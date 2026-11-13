"""Config-based physical episodes using the same policy contract as Gazebo."""
import copy
import math
import select
import sys
import time
import numpy as np
from watercraft.contract import policy_observation
from watercraft.task import metrics, reward


class RealEnv:
    is_real = True

    def __init__(self, settings, operation, seed, camera=None, motors=None, ready=None):
        self.settings = settings
        self.config = settings['real']
        self.task = dict(settings['task'], goal_radius_m=self.config['goal_radius_m'],
                         max_episode_steps=self.config['max_episode_steps'])
        self.goal = np.asarray(self.config['goal'], dtype=np.float32)
        self.control_period_s = self.config['control_period_s']
        self.position_scale_m = settings['policy']['position_scale_m']
        self.goal_radius_m = self.task['goal_radius_m']
        self.max_episode_steps = self.task['max_episode_steps']
        self.np_random = np.random.RandomState(seed)
        self.camera, self.motors, self.ready = camera, motors, ready
        self.count, self.current, self.last_transition = 0, None, None
        self._closed = False
        if not self.config['arm']:
            raise ValueError('Physical runs require arm: true in configs/real.json')
        try:
            if self.camera is None:
                from watercraft.hardware.vision import CameraState
                self.camera = CameraState(self.config)
            if self.motors is None:
                from watercraft.hardware.motors import Motors
                self.motors = Motors(self.config)
        except BaseException:
            self.close()
            raise

    @property
    def unwrapped(self):
        return self

    def _seed(self, seed):
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def _observation(self, state):
        p = self.settings['policy']
        return policy_observation(state['x'], state['y'], state['yaw'], state['vx'], state['vy'],
                                  state['yaw_rate'], *self.goal, position_scale_m=p['position_scale_m'])

    def _metrics(self, state):
        return metrics(state['x'], state['y'], state['yaw'], self.goal, self.goal_radius_m)

    def goal_distance_m(self):
        return self._metrics(self.current)['distance']

    def _verify(self, state):
        if not all(math.isfinite(state[key]) for key in ('x', 'y', 'yaw', 'vx', 'vy', 'yaw_rate', 'timestamp')):
            raise RuntimeError('Non-finite physical state')
        if not state['inside']:
            raise RuntimeError('Boat is outside the calibrated workspace')
        if time.monotonic() - state['timestamp'] > self.config['measurement_timeout_s']:
            raise RuntimeError('Stale physical measurement')

    def reset(self):
        self.stop()
        if self.ready is not None:
            self.ready()
        else:
            print('Motors stopped. Reposition the boat, then press Enter when ready (Q to stop).', flush=True)
            if not select.select([sys.stdin], [], [], self.config['reset_timeout_s'])[0]:
                raise TimeoutError('Operator reset timed out')
            line = sys.stdin.readline()
            if not line or line.strip().lower() == 'q':
                raise KeyboardInterrupt
        self.camera.reset()
        self.current = self.camera.read()
        self._verify(self.current)
        time.sleep(self.control_period_s)
        self.current = self.camera.read()
        self._verify(self.current)
        if math.hypot(self.current['vx'], self.current['vy']) > self.config['reset_max_speed_mps'] or abs(self.current['yaw_rate']) > self.config['reset_max_yaw_rate_radps']:
            raise RuntimeError('Boat must be stationary before starting an episode')
        self.count = 0
        self.last_tick = time.monotonic()
        return self._observation(self.current)

    def step(self, action):
        previous = self.current
        if previous is None:
            raise RuntimeError('Call reset before step')
        try:
            self._verify(previous)
            requested = np.array(action, copy=True)
            channels = self.motors.act(action)
            sent = time.monotonic()
            time.sleep(self.control_period_s)
            current = self.camera.read()
            self._verify(current)
            self.last_tick = time.monotonic()
            self.count += 1
            after = self._metrics(current)
            terminated = after['reached']
            truncated = self.count >= self.max_episode_steps and not terminated
            result_reward = reward(self._metrics(previous), after, self.task)
            self.current = current
            self.last_transition = dict(previous=copy.deepcopy(previous), current=copy.deepcopy(current),
                observation=self._observation(previous).tolist(), next_observation=self._observation(current).tolist(),
                action=requested.tolist(), channels=list(channels), reward=result_reward,
                terminated=bool(terminated), truncated=bool(truncated), command_time=sent)
            if terminated or truncated:
                self.stop()
            return self._observation(current), result_reward, bool(terminated or truncated), {
                'goal_dist_m': after['distance'], 'reached': bool(terminated),
                'TimeLimit.truncated': bool(truncated), 'sim_dt': current['timestamp'] - previous['timestamp'],
                'applied_action': requested, 'channels': channels}
        except BaseException:
            self.stop()
            raise

    def stop(self):
        if self.motors is not None:
            self.motors.stop()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.motors is not None:
                self.motors.close()
        finally:
            if self.camera is not None:
                self.camera.close()
