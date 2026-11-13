"""Configure the existing Gym environment from configs/train.json without rewriting it."""
import time
import numpy as np


def make(settings, seed):
    import gym
    import gym_gazebo  # noqa: F401: registers the existing environment
    from watercraft import contract
    import gym_gazebo.envs.pool_gym_env.pool_gym_v0 as implementation
    implementation.GOAL_RADIUS_M = settings['task']['goal_radius_m']
    implementation.FORWARD_SPEED_SCALE_MPS = settings['policy']['forward_speed_scale_mps']
    implementation.YAW_RATE_SCALE_RADPS = settings['policy']['yaw_rate_scale_radps']
    contract.FORWARD_SPEED_SCALE_MPS = settings['policy']['forward_speed_scale_mps']
    contract.YAW_RATE_SCALE_RADPS = settings['policy']['yaw_rate_scale_radps']
    env = gym.make(settings['sim']['environment_id'])
    try:
        pool = env.unwrapped
        for key in ('control_period_s', 'marker_offset_x_m', 'random_goals', 'goal_distance_range_m',
                    'goal_bounds_m', 'domain_randomization', 'dr_ranges'):
            setattr(pool, key, settings['sim'][key])
        pool.position_scale_m = settings['policy']['position_scale_m']
        for key in ('max_episode_steps', 'distance_weight', 'heading_weight', 'step_penalty', 'goal_bonus'):
            setattr(pool, key, settings['task'][key])
        pool.goal = np.asarray(settings['task']['goal'], dtype=np.float32)
        pool._seed(seed)
        time.sleep(settings['sim']['startup_wait_s'])
        return env
    except BaseException:
        stop(env)
        raise


def stop(env):
    pool = env.unwrapped
    if hasattr(pool, 'left_pub'):
        pool.left_pub.publish(0.0)
        pool.right_pub.publish(0.0)
    if hasattr(pool, 'pause'):
        pool.pause()


def close(env):
    """Stop this session; do not invoke the legacy global killall implementation."""
    try:
        stop(env)
    finally:
        process = getattr(env.unwrapped, '_roslaunch', None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except TimeoutError:
                process.kill()
                process.wait()
            except __import__('subprocess').TimeoutExpired:
                process.kill()
                process.wait()
