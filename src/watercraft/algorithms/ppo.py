"""Proximal Policy Optimization: a Gaussian policy around the deployable actor, plus a critic."""
import copy
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from watercraft.config import path
from watercraft.policy import build, export, torch_state
from watercraft.logging import CsvLog, TensorBoardLog, EPISODE_FIELDS
from watercraft.sim_env import stop

ITERATION_FIELDS = ['iter', 'elapsed_s', 'env_steps', 'loss', 'policy_loss', 'value_loss', 'entropy',
                    'approx_kl', 'clip_fraction', 'reward_mean', 'success_rate', 'mean_final_dist_m',
                    'mean_min_dist_m', 'mean_steps', 'noise_std']
LOG_STD_RANGE = (-3.0, 0.5)  # exploration noise stays between about 0.05 and 1.65


# ---------------------------------------------------------------- 1. model
class ActorCritic(nn.Module):
    def __init__(self, policy_config, initial_log_std):
        super().__init__()
        self.actor = build(policy_config)  # the deployable 6 -> hidden -> 2 network
        self.critic = nn.Sequential(
            nn.Linear(policy_config['observation_size'], policy_config['hidden_size']), nn.ReLU(),
            nn.Linear(policy_config['hidden_size'], 1))
        self.log_std = nn.Parameter(torch.full((policy_config['action_size'],), float(initial_log_std)))

    def distribution(self, observations):
        mean = torch.tanh(self.actor(observations))  # same output the boat uses
        return torch.distributions.Normal(mean, self.log_std.clamp(*LOG_STD_RANGE).exp())

    def value(self, observations):
        return self.critic(observations).squeeze(-1)


def as_tensor(array):
    return torch.as_tensor(np.asarray(array), dtype=torch.float32)


# ---------------------------------------------------------------- 2 + 3. acting and rollout
def collect_rollout(env, model, steps_wanted, goal_radius_m):
    """Run whole episodes until at least steps_wanted steps are stored."""
    pool = env.unwrapped
    data = {key: [] for key in ('observations', 'actions', 'log_probs', 'values', 'rewards', 'next_values', 'ends')}
    episodes = []
    try:
        while len(data['rewards']) < steps_wanted:
            observation = env.reset()
            start = minimum = pool.goal_distance_m()
            total, steps, done = 0.0, 0, False
            while not done:
                with torch.no_grad():
                    tensor = as_tensor(observation)
                    distribution = model.distribution(tensor)
                    action = distribution.sample()
                    log_prob = distribution.log_prob(action).sum(-1)
                    value = model.value(tensor)
                next_observation, reward, done, info = env.step(np.clip(action.numpy(), -1.0, 1.0))
                reached = done and info['goal_dist_m'] <= goal_radius_m
                with torch.no_grad():
                    # Nothing follows a reached goal; a timeout only stops the clock, so ask the critic.
                    next_value = 0.0 if reached else model.value(as_tensor(next_observation)).item()
                data['observations'].append(np.array(observation, copy=True))
                data['actions'].append(action.numpy())  # the unclipped sample the log-prob belongs to
                data['log_probs'].append(log_prob.item())
                data['values'].append(value.item())
                data['rewards'].append(float(reward))
                data['next_values'].append(next_value)
                data['ends'].append(done)
                total += reward
                steps += 1
                minimum = min(minimum, info['goal_dist_m'])
                observation = next_observation
            episodes.append(dict(reward=total, steps=steps, start_dist=start, final_dist=info['goal_dist_m'],
                                 min_dist=minimum, reached=reached, goal=list(map(float, pool.goal))))
            stop(env)
    finally:
        stop(env)
    return {key: np.asarray(value, dtype=bool if key == 'ends' else np.float32) for key, value in data.items()}, episodes


# ---------------------------------------------------------------- 4. advantages
def compute_gae(rewards, values, next_values, ends, gamma, gae_lambda):
    """Return (advantages, returns) by walking backwards through the rollout."""
    advantages = np.zeros(len(rewards), dtype=np.float32)
    advantage = 0.0
    for t in reversed(range(len(rewards))):
        if ends[t]:
            advantage = 0.0  # do not carry anything over from the next episode
        delta = rewards[t] + gamma * next_values[t] - values[t]
        advantage = delta + gamma * gae_lambda * advantage
        advantages[t] = advantage
    return advantages, advantages + values


# ---------------------------------------------------------------- 5. update
def update(model, optimizer, rollout, advantages, returns, config):
    observations, actions = as_tensor(rollout['observations']), as_tensor(rollout['actions'])
    old_log_probs, returns = as_tensor(rollout['log_probs']), as_tensor(returns)
    advantages = as_tensor(advantages)
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    clip = config['clip_range']
    stats = {key: [] for key in ('loss', 'policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_fraction')}
    for _ in range(config['epochs']):
        for index in torch.randperm(len(observations)).split(config['minibatch_size']):
            distribution = model.distribution(observations[index])
            log_probs = distribution.log_prob(actions[index]).sum(-1)
            entropy = distribution.entropy().sum(-1).mean()
            ratio = (log_probs - old_log_probs[index]).exp()
            policy_loss = -torch.min(ratio * advantages[index],
                                     ratio.clamp(1 - clip, 1 + clip) * advantages[index]).mean()
            value_loss = nn.functional.mse_loss(model.value(observations[index]), returns[index])
            loss = policy_loss + config['value_coef'] * value_loss - config['entropy_coef'] * entropy
            optimizer.zero_grad()
            loss.backward()
            # Clip the two networks separately so a large critic gradient cannot shrink the actor's step.
            nn.utils.clip_grad_norm_(list(model.actor.parameters()) + [model.log_std], config['max_grad_norm'])
            nn.utils.clip_grad_norm_(model.critic.parameters(), config['max_grad_norm'])
            optimizer.step()
            with torch.no_grad():
                for key, value in (('loss', loss), ('policy_loss', policy_loss), ('value_loss', value_loss),
                                   ('entropy', entropy), ('approx_kl', (old_log_probs[index] - log_probs).mean()),
                                   ('clip_fraction', ((ratio - 1).abs() > clip).float().mean())):
                    stats[key].append(value.item())
    return {key: float(np.mean(values)) for key, values in stats.items()}


# ---------------------------------------------------------------- 6. outer loop
def save_checkpoint(filename, model, optimizer, state, settings):
    payload = dict(state, model=model.state_dict(), optimizer=optimizer.state_dict(), settings=settings,
                   numpy_rng=np.random.get_state(), python_rng=random.getstate(), torch_rng=torch.get_rng_state())
    temporary = Path(filename).with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(filename)


def save_actor(filename, weights, settings):
    torch.save(weights, filename)
    if settings['policy']['export_numpy']:
        export(filename)


def train(settings, env, directory, resume=None):
    config, options = settings['algorithm'], settings['train']
    directory = Path(directory)
    checkpoint_dir = directory / 'checkpoints'
    pool = env.unwrapped
    goal_radius_m = getattr(pool, 'goal_radius_m', settings['task']['goal_radius_m'])
    model = ActorCritic(settings['policy'], config['initial_log_std'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    state = {'iteration': 0, 'episode_no': 0, 'total_goals': 0, 'elapsed_s': 0.0, 'env_steps': 0,
             'best_score': None, 'success_streak': 0, 'stop_reason': 'running'}
    if resume:
        state.update({key: resume[key] for key in state})
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        np.random.set_state(resume['numpy_rng'])
        random.setstate(resume['python_rng'])
        torch.set_rng_state(resume['torch_rng'])
        if resume.get('environment_rng') is not None:
            if hasattr(pool.np_random, 'set_state'):
                pool.np_random.set_state(resume['environment_rng'])
            else:
                pool.np_random.bit_generator.state = resume['environment_rng']
    elif options['initialize_from']:
        model.actor.load_state_dict(torch_state(path(options['initialize_from'])))
    state['stop_reason'] = 'running'
    episode_log = CsvLog(directory / 'training.csv', EPISODE_FIELDS, append=bool(resume))
    iteration_log = CsvLog(directory / 'iterations.csv', ITERATION_FIELDS, append=bool(resume))
    tb = TensorBoardLog(str(directory / 'tensorboard'))
    started, previous_elapsed = time.monotonic(), state['elapsed_s']
    try:
        while True:
            iteration = state['iteration']
            noise = float(model.log_std.clamp(*LOG_STD_RANGE).exp().mean())
            collected_actor = copy.deepcopy(model.actor.state_dict())  # the actor these episodes belong to
            rollout, episodes = collect_rollout(env, model, config['steps_per_iteration'], goal_radius_m)
            state['elapsed_s'] = previous_elapsed + time.monotonic() - started
            state['env_steps'] += len(rollout['rewards'])
            for episode in episodes:
                episode_log.write(iter=iteration, episode=state['episode_no'], total_reward=episode['reward'],
                    steps=episode['steps'], start_dist_m=episode['start_dist'], final_dist_m=episode['final_dist'],
                    min_dist_m=episode['min_dist'], reached=int(episode['reached']), goal_x_m=episode['goal'][0],
                    goal_y_m=episode['goal'][1], noise_std=noise, elapsed_s=state['elapsed_s'])
                state['episode_no'] += 1
                state['total_goals'] += int(episode['reached'])
            success = float(np.mean([e['reached'] for e in episodes]))
            final_distance = float(np.mean([e['final_dist'] for e in episodes]))
            score = (success, -final_distance)
            if state['best_score'] is None or score > state['best_score']:
                state['best_score'] = score
                save_actor(checkpoint_dir / 'best.pth', collected_actor, settings)
            advantages, returns = compute_gae(rollout['rewards'] * config['reward_scale'], rollout['values'],
                                              rollout['next_values'], rollout['ends'],
                                              config['gamma'], config['gae_lambda'])
            row = dict(update(model, optimizer, rollout, advantages, returns, config),
                iter=iteration, elapsed_s=state['elapsed_s'], env_steps=state['env_steps'],
                reward_mean=float(np.mean([e['reward'] for e in episodes])), success_rate=success,
                mean_final_dist_m=final_distance, mean_min_dist_m=float(np.mean([e['min_dist'] for e in episodes])),
                mean_steps=float(np.mean([e['steps'] for e in episodes])), noise_std=noise)
            iteration_log.write(**row)
            tb.write(iteration, row)
            print(f"Iteration {iteration}: success={success:.0%}, distance={final_distance:.3f}m, "
                  f"kl={row['approx_kl']:.4f}, std={noise:.2f}, steps={state['env_steps']}")
            state['iteration'] += 1
            state['success_streak'] = state['success_streak'] + 1 if success >= config['stop_success'] else 0
            if state['success_streak'] >= config['stop_iterations']:
                state['stop_reason'] = 'solved'
            elif state['elapsed_s'] >= options['minutes'] * 60:
                state['stop_reason'] = 'time_limit'
            elif options['max_iterations'] is not None and state['iteration'] >= options['max_iterations']:
                state['stop_reason'] = 'iteration_limit'
            rng = getattr(pool, 'np_random', None)
            state['environment_rng'] = rng.get_state() if hasattr(rng, 'get_state') else (
                copy.deepcopy(rng.bit_generator.state) if rng is not None else None)
            save_checkpoint(checkpoint_dir / 'latest.pt', model, optimizer, state, settings)
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
        save_actor(checkpoint_dir / 'final.pth', model.actor.state_dict(), settings)
        summary = {key: state[key] for key in ('iteration', 'episode_no', 'total_goals', 'elapsed_s', 'env_steps', 'stop_reason')}
        if 'fault' in state:
            summary['fault'] = state['fault']
        (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
    return state
