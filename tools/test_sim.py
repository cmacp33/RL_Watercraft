"""Run the configured random-action simulator diagnostic."""
import time
from watercraft.config import resolve, load
from watercraft.envs import make, close

def main():
    settings = resolve('train')
    settings['train']['environment'] = 'sim'
    env = make(settings, 'train', settings['train']['seed'])
    try:
        env.reset()
        deadline = time.monotonic() + load('tools')['sim_test_seconds']
        while time.monotonic() < deadline:
            observation, reward, done, info = env.step(env.action_space.sample())
            print('Reward:', reward, 'Distance:', info['goal_dist_m'])
            if done:
                env.reset()
    finally:
        close(env)

if __name__ == '__main__':
    main()
