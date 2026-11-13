"""The existing elite-episode imitation algorithm, with config-driven run handling."""
import copy
import random
import time
from collections import namedtuple
from pathlib import Path
import numpy as np
import torch
from watercraft.config import load
from watercraft.policy import Net, export, build, TorchPolicy, torch_state
from watercraft.logging import CsvLog, TensorBoardLog, EPISODE_FIELDS, ITERATION_FIELDS
from watercraft.sim_env import stop

_CONFIG = load('algorithm')
REF_SPEED_MPS = _CONFIG['reference_speed_mps']
Episode = namedtuple('Episode', 'reward steps start_dist final_dist min_dist reached goal')
EpisodeStep = namedtuple('EpisodeStep', 'observation action')


def episode_score(ep, control_period_s, reference_speed_mps=REF_SPEED_MPS):
    if ep.reached:
        speed = ep.start_dist / max(1, len(ep.steps)) / control_period_s
        return 1.0 + min(1.0, speed / reference_speed_mps)
    return (ep.start_dist - ep.final_dist) / max(ep.start_dist, 1e-6)


def select_elites(scored_batch, percentile):
    bound = np.percentile([score for score, _ in scored_batch], percentile)
    return [(score, ep) for score, ep in scored_batch if score >= bound and score > 0], bound


def elite_tensors(memory):
    return (torch.as_tensor(np.asarray([s.observation for _, e in memory for s in e.steps]), dtype=torch.float32),
            torch.as_tensor(np.asarray([s.action for _, e in memory for s in e.steps]), dtype=torch.float32))


def collect_episode(env, net, noise_std):
    pool = env.unwrapped
    observation = env.reset()
    start = minimum = pool.goal_distance_m()
    total, steps, durations = 0.0, [], []
    done = False
    try:
        while not done:
            action = TorchPolicy(net).act(observation)
            action = np.clip(action + np.random.normal(0, noise_std, size=action.shape), -1, 1)
            next_observation, reward, done, info = env.step(action)
            steps.append(EpisodeStep(np.array(observation, copy=True), np.array(action, copy=True)))
            total += reward
            minimum = min(minimum, info['goal_dist_m'])
            durations.append(info.get('sim_dt', pool.control_period_s))
            transition_log = getattr(pool, 'transition_log', None)
            if transition_log is not None and pool.last_transition is not None:
                import json
                transition_log.write(episode=pool.recording_episode, step=len(steps),
                                     transition=json.dumps(pool.last_transition))
            observation = next_observation
        radius = getattr(pool, 'goal_radius_m', load('task')['goal_radius_m'])
        final = pool.goal_distance_m()
        return Episode(total, steps, start, final, minimum, final <= radius, list(map(float, pool.goal))), durations
    finally:
        stop(env)


def iterate_batches(env, net, batch_size, noise):
    while True:
        batch, durations = [], []
        for _ in range(batch_size):
            episode, times = collect_episode(env, net, noise[0])
            batch.append(episode)
            durations.extend(times)
        yield batch, float(np.mean(durations))


def save_checkpoint(filename, model, optimizer, state, settings):
    payload = dict(state, model=model.state_dict(), optimizer=optimizer.state_dict(), settings=settings,
                   numpy_rng=np.random.get_state(), python_rng=random.getstate(), torch_rng=torch.get_rng_state())
    temporary = Path(filename).with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(filename)


def train(settings, env, directory, resume=None):
    config, options = settings['algorithm'], settings['train']
    directory = Path(directory)
    net = build(settings['policy'])
    optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'])
    state = {'iteration': 0, 'episode_no': 0, 'total_goals': 0, 'elapsed_s': 0.0,
             'memory': [], 'best_score': None, 'success_streak': 0, 'stop_reason': 'running'}
    if resume:
        state.update({k: resume[k] for k in state})
        net.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        np.random.set_state(resume['numpy_rng'])
        random.setstate(resume['python_rng'])
        torch.set_rng_state(resume['torch_rng'])
        if 'environment_rng' in resume and resume['environment_rng'] is not None:
            pool_rng = env.unwrapped.np_random
            if hasattr(pool_rng, 'set_state'):
                pool_rng.set_state(resume['environment_rng'])
            else:
                pool_rng.bit_generator.state = resume['environment_rng']
    elif options['initialize_from']:
        net.load_state_dict(torch_state(__import__('watercraft.config', fromlist=['path']).path(options['initialize_from'])))
    state['stop_reason'] = 'running'
    checkpoint_dir = directory / 'checkpoints'
    # Append only for an explicit resume; never replace an earlier history.
    episode_log = CsvLog(directory / 'training.csv', EPISODE_FIELDS, append=bool(resume))
    iteration_log = CsvLog(directory / 'iterations.csv', ITERATION_FIELDS, append=bool(resume))
    tb = TensorBoardLog(str(directory / 'tensorboard'))
    if getattr(env.unwrapped, 'is_real', False) and hasattr(env.unwrapped, 'last_transition'):
        env.unwrapped.transition_log = CsvLog(directory / 'transitions.csv', ['episode', 'step', 'transition'], append=bool(resume))
    started, previous_elapsed = time.monotonic(), state['elapsed_s']
    try:
        while True:
            iteration = state['iteration']
            fraction = min(1.0, iteration / config['noise_decay_iterations'])
            noise = config['noise_start'] + fraction * (config['noise_end'] - config['noise_start'])
            # Freeze the exact actor that generates these selection scores.
            collected_weights = copy.deepcopy(net.state_dict())
            batch, durations = [], []
            for batch_index in range(config['batch_size']):
                env.unwrapped.recording_episode = state['episode_no'] + batch_index
                episode, times = collect_episode(env, net, noise)
                batch.append(episode)
                durations.extend(times)
            state['elapsed_s'] = previous_elapsed + time.monotonic() - started
            scored = [(episode_score(e, env.unwrapped.control_period_s, config['reference_speed_mps']), e) for e in batch]
            for score, episode in scored:
                episode_log.write(iter=iteration, episode=state['episode_no'], total_reward=episode.reward,
                    steps=len(episode.steps), start_dist_m=episode.start_dist, final_dist_m=episode.final_dist,
                    min_dist_m=episode.min_dist, reached=int(episode.reached), goal_x_m=episode.goal[0],
                    goal_y_m=episode.goal[1], score=score, noise_std=noise, elapsed_s=state['elapsed_s'])
                state['episode_no'] += 1
                state['total_goals'] += int(episode.reached)
            success = float(np.mean([e.reached for e in batch]))
            final_distance = float(np.mean([e.final_dist for e in batch]))
            score = (success, -final_distance)
            if state['best_score'] is None or score > state['best_score']:
                state['best_score'] = score
                torch.save(collected_weights, checkpoint_dir / 'best.pth')
                if settings['policy']['export_numpy']:
                    export(checkpoint_dir / 'best.pth')
            elites, bound = select_elites(scored, config['percentile'])
            state['memory'] = sorted(state['memory'] + elites, key=lambda item: item[0], reverse=True)[:config['elite_memory']]
            loss = float('nan')
            if state['memory']:
                observations, actions = elite_tensors(state['memory'])
                for _ in range(config['train_steps_per_iteration']):
                    optimizer.zero_grad()
                    loss_tensor = torch.nn.functional.mse_loss(torch.tanh(net(observations)), actions)
                    loss_tensor.backward()
                    optimizer.step()
                loss = loss_tensor.item()
            rewards = [e.reward for e in batch]
            row = dict(iter=iteration, elapsed_s=state['elapsed_s'], loss=loss,
                reward_mean=float(np.mean(rewards)), reward_bound=float(np.percentile(rewards, config['percentile'])),
                success_rate=success, mean_final_dist_m=final_distance,
                mean_min_dist_m=float(np.mean([e.min_dist for e in batch])), mean_steps=float(np.mean([len(e.steps) for e in batch])),
                train_steps_kept=sum(len(e.steps) for _, e in state['memory']), noise_std=noise,
                score_mean=float(np.mean([v for v, _ in scored])), score_bound=float(bound), new_elites=len(elites),
                memory_size=len(state['memory']), memory_success=float(np.mean([e.reached for _, e in state['memory']])) if state['memory'] else 0.0)
            iteration_log.write(**row)
            tb.write(iteration, row)
            print(f"Iteration {iteration}: success={success:.0%}, distance={final_distance:.3f}m, loss={loss:.4f}")
            state['iteration'] += 1
            state['success_streak'] = state['success_streak'] + 1 if success >= config['stop_success'] else 0
            if state['success_streak'] >= config['stop_iterations']:
                state['stop_reason'] = 'solved'
            elif state['elapsed_s'] >= options['minutes'] * 60:
                state['stop_reason'] = 'time_limit'
            elif options['max_iterations'] is not None and state['iteration'] >= options['max_iterations']:
                state['stop_reason'] = 'iteration_limit'
            rng = getattr(env.unwrapped, 'np_random', None)
            state['environment_rng'] = rng.get_state() if hasattr(rng, 'get_state') else (copy.deepcopy(rng.bit_generator.state) if rng is not None else None)
            save_checkpoint(checkpoint_dir / 'latest.pt', net, optimizer, state, settings)
            if state['stop_reason'] != 'running':
                break
    except KeyboardInterrupt:
        state['stop_reason'] = 'interrupted'
    except BaseException as error:
        state['stop_reason'] = 'fault'
        state['fault'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        stop(env)
        episode_log.close()
        iteration_log.close()
        tb.close()
        transition_log = getattr(env.unwrapped, 'transition_log', None)
        if transition_log is not None:
            transition_log.close()
        # Do not advertise an incomplete interrupted rollout as an exactly resumable boundary.
        torch.save(net.state_dict(), checkpoint_dir / 'final.pth')
        if settings['policy']['export_numpy']:
            export(checkpoint_dir / 'final.pth')
        import json
        summary = {k: state[k] for k in ('iteration', 'episode_no', 'total_goals', 'elapsed_s', 'stop_reason')}
        if 'fault' in state:
            summary['fault'] = state['fault']
        (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
    return state
