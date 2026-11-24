"""Soft Actor-Critic: an off-policy, entropy-regularised actor around the deployable network.

Ported from Nora's training_scripts/gazebo_pool_v0_sac.py (branch nora/SAC) into the config-driven
layout. SAC learns from every stored step through the Bellman equation

    Q(s, a) = r + gamma * (1 - terminal) * [ min(Q1', Q2')(s', a') - alpha * log pi(a'|s') ]

and reuses each transition many times from a replay buffer. The policy mean is exactly the
6 -> hidden -> 2 network the boat runs (tanh of its output), so best/final .pth and .npz files work
with evaluate and deploy unchanged.

The algorithm section of configs/train.json for SAC (train.algorithm = "sac"):
    learning_rate 3e-4, gamma 0.99, tau 0.005, batch_size 256, buffer_size 300000,
    start_steps 2000, updates_per_step 1, critic_hidden 256, initial_alpha 0.2,
    target_entropy null (= -action_size), reward_scale 0.1, episodes_per_iteration 8,
    eval_every 5, eval_episodes 10, eval_seed 12345, stop_success 0.9, stop_iterations 3
    ("solved" needs the evaluation and that iteration's training success both >= stop_success)
"""
import copy
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from watercraft.config import path
from watercraft.policy import build, export, torch_state
from watercraft.logging import CsvLog, TensorBoardLog, EPISODE_FIELDS
from watercraft.sim_env import stop

ITERATION_FIELDS = ['iter', 'elapsed_s', 'env_steps', 'loss', 'actor_loss', 'alpha', 'q_mean', 'noise_std',
                    'reward_mean', 'success_rate', 'mean_final_dist_m', 'mean_min_dist_m', 'mean_steps',
                    'buffer_size', 'eval_success_rate', 'eval_final_dist_m', 'eval_median_steps_to_goal']
# eval_* are filled only on iterations that ran an evaluation (every eval_every), blank otherwise.
LOG_STD_RANGE = (-5.0, 2.0)


# ---------------------------------------------------------------- 1. model
class Actor(nn.Module):
    """Gaussian policy squashed by tanh; `mean_net` is the deployable network."""

    def __init__(self, policy_config):
        super().__init__()
        self.mean_net = build(policy_config)
        self.log_std_head = nn.Linear(policy_config['hidden_size'], policy_config['action_size'])

    def forward(self, observations):
        layers = self.mean_net.net
        hidden = layers[1](layers[0](observations))  # Linear + ReLU, shared with the std head
        return layers[2](hidden), self.log_std_head(hidden).clamp(*LOG_STD_RANGE)

    def sample(self, observations):
        """Random action in [-1, 1], its log-probability, and the std (all differentiable)."""
        mean, log_std = self(observations)
        std = log_std.exp()
        u = mean + std * torch.randn_like(mean)  # reparameterisation trick
        log_prob = (-0.5 * ((u - mean) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        # tanh correction log(1 - tanh(u)^2), written so it cannot overflow
        log_prob = log_prob - (2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))).sum(-1)
        return torch.tanh(u), log_prob, std

    def deterministic(self, observations):
        return torch.tanh(self(observations)[0])  # exactly what the boat runs


class Critic(nn.Module):
    """Q(s, a): expected discounted future reward after taking action a in state s."""

    def __init__(self, policy_config, hidden):
        super().__init__()
        inputs = policy_config['observation_size'] + policy_config['action_size']
        self.net = nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, observations, actions):
        return self.net(torch.cat([observations, actions], dim=-1)).squeeze(-1)


class ReplayBuffer:
    """Fixed-size store of transitions; the oldest are overwritten once it is full."""
    KEYS = ('observations', 'actions', 'rewards', 'next_observations', 'terminals')

    def __init__(self, size, observation_size, action_size):
        self.data = {'observations': np.zeros((size, observation_size), np.float32),
                     'actions': np.zeros((size, action_size), np.float32),
                     'rewards': np.zeros(size, np.float32),
                     'next_observations': np.zeros((size, observation_size), np.float32),
                     'terminals': np.zeros(size, np.float32)}
        self.size, self.count, self.index = size, 0, 0

    def add(self, observation, action, reward, next_observation, terminal):
        for key, value in zip(self.KEYS, (observation, action, reward, next_observation, float(terminal))):
            self.data[key][self.index] = value
        self.index = (self.index + 1) % self.size
        self.count = min(self.count + 1, self.size)

    def sample(self, batch_size):
        index = np.random.randint(0, self.count, size=batch_size)
        return [torch.from_numpy(self.data[key][index]) for key in self.KEYS]

    def state_dict(self):
        return dict(count=self.count, index=self.index, **{k: v[:self.count].copy() for k, v in self.data.items()})

    def load_state_dict(self, state):
        self.count, self.index = state['count'], state['index']
        for key in self.KEYS:
            self.data[key][:self.count] = state[key]


class SAC:
    def __init__(self, policy_config, config):
        self.config = config
        self.actor = Actor(policy_config)
        self.q1, self.q2 = (Critic(policy_config, config['critic_hidden']) for _ in range(2))
        self.q1_target, self.q2_target = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.log_alpha = torch.tensor(math.log(config['initial_alpha']), requires_grad=True)
        target = config.get('target_entropy')
        self.target_entropy = -float(policy_config['action_size']) if target is None else float(target)
        self.critic_params = list(self.q1.parameters()) + list(self.q2.parameters())
        lr = config['learning_rate']
        self.optimizers = {'actor': torch.optim.Adam(self.actor.parameters(), lr=lr),
                           'critic': torch.optim.Adam(self.critic_params, lr=lr),
                           'alpha': torch.optim.Adam([self.log_alpha], lr=lr)}

    def act(self, observation, deterministic=False):
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(observation), dtype=torch.float32).unsqueeze(0)
            action = self.actor.deterministic(tensor) if deterministic else self.actor.sample(tensor)[0]
        return action.numpy()[0]

    def update(self, buffer):
        config = self.config
        observations, actions, rewards, next_observations, terminals = buffer.sample(config['batch_size'])
        rewards = rewards * config['reward_scale']
        alpha = self.log_alpha.exp().detach()

        # 1. critics regress onto the Bellman target; only a reached goal ends the future.
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_observations)
            next_q = torch.min(self.q1_target(next_observations, next_actions),
                               self.q2_target(next_observations, next_actions))
            target = rewards + config['gamma'] * (1.0 - terminals) * (next_q - alpha * next_log_probs)
        q1 = self.q1(observations, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(self.q2(observations, actions), target)
        self.optimizers['critic'].zero_grad()
        critic_loss.backward()
        self.optimizers['critic'].step()

        # 2. actor: pick actions the critics rate highly while staying random; critics frozen.
        for parameter in self.critic_params:
            parameter.requires_grad_(False)
        new_actions, log_probs, std = self.actor.sample(observations)
        actor_loss = (alpha * log_probs - torch.min(self.q1(observations, new_actions),
                                                    self.q2(observations, new_actions))).mean()
        self.optimizers['actor'].zero_grad()
        actor_loss.backward()
        self.optimizers['actor'].step()
        for parameter in self.critic_params:
            parameter.requires_grad_(True)

        # 3. alpha rises when the policy is less random than the target entropy, falls otherwise.
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy).mean())
        self.optimizers['alpha'].zero_grad()
        alpha_loss.backward()
        self.optimizers['alpha'].step()

        # 4. target critics slowly follow the critics.
        with torch.no_grad():
            for network, target_network in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
                for parameter, target_parameter in zip(network.parameters(), target_network.parameters()):
                    target_parameter.mul_(1.0 - config['tau']).add_(config['tau'] * parameter)

        return {'loss': critic_loss.item(), 'actor_loss': actor_loss.item(), 'alpha': self.log_alpha.exp().item(),
                'q_mean': q1.mean().item(), 'noise_std': std.mean().item()}

    def state_dict(self):
        return {'actor': self.actor.state_dict(), 'q1': self.q1.state_dict(), 'q2': self.q2.state_dict(),
                'q1_target': self.q1_target.state_dict(), 'q2_target': self.q2_target.state_dict(),
                'log_alpha': self.log_alpha.detach().clone()}

    def load_state_dict(self, state):
        for name in ('actor', 'q1', 'q2', 'q1_target', 'q2_target'):
            getattr(self, name).load_state_dict(state[name])
        with torch.no_grad():
            self.log_alpha.copy_(state['log_alpha'])


# ---------------------------------------------------------------- 2. evaluation
def evaluate(env, agent, episodes, seed, goal_radius_m):
    """Noise-free episodes on the same goals/conditions every call; not added to the buffer.

    Returns (success rate, mean final distance, median steps over the episodes that reached the goal).
    Steps count only reached episodes, like median_steps_to_goal in watercraft.evaluate, so a timeout
    (max_episode_steps) does not inflate them; it is nan when no episode reached the goal.
    """
    pool = env.unwrapped
    training_rng = getattr(pool, 'np_random', None)
    pool.np_random = np.random.RandomState(seed)
    results = []
    try:
        for _ in range(episodes):
            observation, done, steps = env.reset(), False, 0
            while not done:
                observation, _, done, info = env.step(agent.act(observation, deterministic=True))
                steps += 1
            results.append((info['reached'], info['goal_dist_m'], steps))
            stop(env)
    finally:
        pool.np_random = training_rng
        stop(env)
    reached = [steps for ok, _, steps in results if ok]
    return (float(np.mean([r[0] for r in results])), float(np.mean([r[1] for r in results])),
            float(np.median(reached)) if reached else float('nan'))


def reached_any(median_steps):
    return not math.isnan(median_steps)


# ---------------------------------------------------------------- 3. outer loop
def save_checkpoint(filename, agent, buffer, state, settings):
    payload = dict(state, model=agent.state_dict(),
                   optimizer={k: o.state_dict() for k, o in agent.optimizers.items()},
                   buffer=buffer.state_dict(), settings=settings, numpy_rng=np.random.get_state(),
                   python_rng=random.getstate(), torch_rng=torch.get_rng_state())
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
    action_size = settings['policy']['action_size']
    agent = SAC(settings['policy'], config)
    buffer = ReplayBuffer(config['buffer_size'], settings['policy']['observation_size'], action_size)
    state = {'iteration': 0, 'episode_no': 0, 'total_goals': 0, 'elapsed_s': 0.0, 'env_steps': 0,
             'best_score': None, 'success_streak': 0, 'stop_reason': 'running',
             'last_eval': [float('nan')] * 3, 'stats': {}}
    if resume:
        state.update({key: resume[key] for key in state if key in resume})
        agent.load_state_dict(resume['model'])
        for key, optimizer in agent.optimizers.items():
            optimizer.load_state_dict(resume['optimizer'][key])
        buffer.load_state_dict(resume['buffer'])
        np.random.set_state(resume['numpy_rng'])
        random.setstate(resume['python_rng'])
        torch.set_rng_state(resume['torch_rng'])
        if resume.get('environment_rng') is not None:
            if hasattr(pool.np_random, 'set_state'):
                pool.np_random.set_state(resume['environment_rng'])
            else:
                pool.np_random.bit_generator.state = resume['environment_rng']
    elif options['initialize_from']:
        agent.actor.mean_net.load_state_dict(torch_state(path(options['initialize_from'])))
    state['stop_reason'] = 'running'
    nan_stats = {key: float('nan') for key in ('loss', 'actor_loss', 'alpha', 'q_mean', 'noise_std')}
    stats = dict(nan_stats, **state['stats'])
    episode_log = CsvLog(directory / 'training.csv', EPISODE_FIELDS, append=bool(resume))
    iteration_log = CsvLog(directory / 'iterations.csv', ITERATION_FIELDS, append=bool(resume))
    tb = TensorBoardLog(str(directory / 'tensorboard'))
    started, previous_elapsed = time.monotonic(), state['elapsed_s']
    try:
        while True:
            iteration = state['iteration']
            episodes = []
            for _ in range(config['episodes_per_iteration']):
                observation = env.reset()
                start = minimum = pool.goal_distance_m()
                total, steps, done = 0.0, 0, False
                while not done:
                    if state['env_steps'] < config['start_steps']:
                        action = np.random.uniform(-1.0, 1.0, size=action_size).astype(np.float32)
                    else:
                        action = agent.act(observation)
                    next_observation, reward, done, info = env.step(action)
                    reached = done and info['reached']  # inside the radius for task.goal_hold_steps in a row
                    # A timeout only stops the clock, so it keeps the future term (not terminal).
                    buffer.add(observation, action, reward, next_observation, reached)
                    observation = next_observation
                    total += reward
                    steps += 1
                    state['env_steps'] += 1
                    minimum = min(minimum, info['goal_dist_m'])
                    # Physics is paused between steps, so learning here costs no simulated time.
                    if state['env_steps'] >= config['start_steps'] and buffer.count >= config['batch_size']:
                        for _ in range(config['updates_per_step']):
                            stats = agent.update(buffer)
                stop(env)
                state['elapsed_s'] = previous_elapsed + time.monotonic() - started
                episode = dict(reward=total, steps=steps, start_dist=start, final_dist=info['goal_dist_m'],
                               min_dist=minimum, reached=reached, goal=list(map(float, pool.goal)))
                episodes.append(episode)
                episode_log.write(iter=iteration, episode=state['episode_no'], total_reward=total, steps=steps,
                    start_dist_m=start, final_dist_m=episode['final_dist'], min_dist_m=minimum,
                    reached=int(reached), goal_x_m=episode['goal'][0], goal_y_m=episode['goal'][1],
                    noise_std=stats['noise_std'], elapsed_s=state['elapsed_s'])
                state['episode_no'] += 1
                state['total_goals'] += int(reached)

            evaluated = (iteration % config['eval_every'] == config['eval_every'] - 1
                         and state['env_steps'] >= config['start_steps'])
            if evaluated:
                state['last_eval'] = list(evaluate(env, agent, config['eval_episodes'], config['eval_seed'],
                                                   goal_radius_m))
                success, distance, eval_steps = state['last_eval']
                # Best = noise-free task performance: success, then closeness, then speed.
                score = (success, -round(distance, 3), -eval_steps if reached_any(eval_steps) else -math.inf)
                if state['best_score'] is None or tuple(score) > tuple(state['best_score']):
                    state['best_score'] = score
                    save_actor(checkpoint_dir / 'best.pth', copy.deepcopy(agent.actor.mean_net.state_dict()),
                               settings)
            state['stats'] = stats
            state['elapsed_s'] = previous_elapsed + time.monotonic() - started
            success_rate = float(np.mean([e['reached'] for e in episodes]))
            if evaluated:
                # Solved needs the noise-free evaluation AND this iteration's training episodes to pass:
                # a few evaluation episodes alone once hid a collapse (Nora's run, iterations 6-9).
                passed = min(success, success_rate) >= config['stop_success']
                state['success_streak'] = state['success_streak'] + 1 if passed else 0
            final_distance = float(np.mean([e['final_dist'] for e in episodes]))
            row = dict(stats, iter=iteration, elapsed_s=state['elapsed_s'], env_steps=state['env_steps'],
                       reward_mean=float(np.mean([e['reward'] for e in episodes])), success_rate=success_rate,
                       mean_final_dist_m=final_distance,
                       mean_min_dist_m=float(np.mean([e['min_dist'] for e in episodes])),
                       mean_steps=float(np.mean([e['steps'] for e in episodes])), buffer_size=buffer.count)
            if evaluated:
                row.update(eval_success_rate=state['last_eval'][0], eval_final_dist_m=state['last_eval'][1],
                           eval_median_steps_to_goal=state['last_eval'][2])
            iteration_log.write(**row)
            tb.write(iteration, row)
            print(f"Iteration {iteration}: success={success_rate:.0%}, distance={final_distance:.3f}m, "
                  f"alpha={stats['alpha']:.3f}, std={stats['noise_std']:.2f}, steps={state['env_steps']}"
                  + ('  (random actions)' if state['env_steps'] <= config['start_steps'] else ''))
            if evaluated:
                print(f"   eval: success={state['last_eval'][0]:.0%}, distance={state['last_eval'][1]:.3f}m, "
                      f"median steps to goal={state['last_eval'][2]:.0f}")
            state['iteration'] += 1
            if state['success_streak'] >= config['stop_iterations']:
                state['stop_reason'] = 'solved'
            elif state['elapsed_s'] >= options['minutes'] * 60:
                state['stop_reason'] = 'time_limit'
            elif options['max_iterations'] is not None and state['iteration'] >= options['max_iterations']:
                state['stop_reason'] = 'iteration_limit'
            rng = getattr(pool, 'np_random', None)
            state['environment_rng'] = rng.get_state() if hasattr(rng, 'get_state') else (
                copy.deepcopy(rng.bit_generator.state) if rng is not None else None)
            save_checkpoint(checkpoint_dir / 'latest.pt', agent, buffer, state, settings)
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
        save_actor(checkpoint_dir / 'final.pth', agent.actor.mean_net.state_dict(), settings)
        if not (checkpoint_dir / 'best.pth').exists():
            save_actor(checkpoint_dir / 'best.pth', agent.actor.mean_net.state_dict(), settings)
        summary = {key: state[key] for key in ('iteration', 'episode_no', 'total_goals', 'elapsed_s', 'env_steps',
                                               'stop_reason')}
        if 'fault' in state:
            summary['fault'] = state['fault']
        (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
    return state
