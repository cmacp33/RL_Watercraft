"""Integration checks without opening ROS, cameras, serial ports or motors."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from watercraft.config import resolve, ROOT, run_path
from watercraft.policy import build, load, torch_load
from watercraft.logging import create_run, CsvLog
from watercraft.algorithms.xentropy import train
from watercraft.evaluate import evaluate, EVAL_FIELDS


class FakePool:
    def __init__(self):
        self.goal = np.array([1., 0.])
        self.goal_radius_m = 0.1
        self.control_period_s = 0.1
        self.np_random = np.random.RandomState(42)
        self.stopped = 0
    @property
    def unwrapped(self):
        return self
    def reset(self):
        self.distance, self.steps = 1.0, 0
        return np.zeros(6, dtype=np.float32)
    def goal_distance_m(self):
        return self.distance
    def step(self, action):
        self.steps += 1
        self.distance = max(0.0, 1.0 - self.steps / 2)
        return np.zeros(6, dtype=np.float32), 1., self.steps >= 2, {'goal_dist_m': self.distance, 'sim_dt': .1}
    def pause(self):  # what watercraft.sim_env.stop calls on the Gazebo environment
        self.stopped += 1


class RefactorTests(unittest.TestCase):
    def settings(self):
        c = resolve('train')
        # train.json holds whichever algorithm was run last, so these tests carry their own settings.
        c['train']['algorithm'] = 'xentropy'
        c['algorithm'] = dict(batch_size=2, percentile=70, learning_rate=0.01, noise_start=0.5, noise_end=0.1,
                              noise_decay_iterations=30, elite_memory=50, train_steps_per_iteration=1,
                              reference_speed_mps=2.0, stop_success=2, stop_iterations=3)
        c['train'].update(max_iterations=1, auto_plot=False)
        return c

    def test_contract_and_historical_exports(self):
        from watercraft.contract import NumpyPolicy
        observations = np.random.RandomState(17).uniform(-1, 1, (100, 6)).astype(np.float32)
        pairs = list((ROOT / 'runs').glob('*/*.npz'))
        count = 0
        for npz in pairs:
            pth = npz.with_suffix('.pth')
            if not pth.exists():
                continue
            actor = load(pth, resolve('train')['policy'])
            numpy_actor = NumpyPolicy(npz)
            for obs in observations:
                np.testing.assert_allclose(actor.act(obs), numpy_actor.act(obs), atol=1e-6, rtol=1e-6)
            count += 1
        self.assertGreater(count, 0)

    def test_training_resume_evaluation_and_graphs(self):
        from watercraft.results import plot_run, plot_evaluation
        c = self.settings()
        env = FakePool()
        with tempfile.TemporaryDirectory() as temp:
            with patch('watercraft.logging.run_path', return_value=Path(temp) / 'test'):
                directory = create_run(c)
                with self.assertRaises(FileExistsError):
                    create_run(c)
            result = train(c, env, directory)
            self.assertEqual(result['iteration'], 1)
            checkpoint = directory / 'checkpoints/latest.pt'
            resumed = torch_load(checkpoint)
            c['train']['max_iterations'] = 2
            result = train(c, env, directory, resumed)
            self.assertEqual(result['iteration'], 2)
            self.assertEqual(len((directory / 'training.csv').read_text().splitlines()), 5)
            # The selected best actor is the pre-update actor scored by its batch.
            actor = load(directory / 'checkpoints/best.npz', c['policy'])
            c['evaluate'] = resolve('evaluate')['evaluate']
            c['evaluate'].update(episodes=2, auto_plot=False)
            filename = directory / 'evaluations/sim/test.csv'
            log = CsvLog(filename, EVAL_FIELDS)
            try:
                rows = evaluate(c, env, actor, log)
            finally:
                log.close()
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row['reached'] for row in rows))
            plot_run(directory)
            plot_evaluation(filename)
            self.assertTrue((directory / 'graphs/training.png').is_file())
            self.assertTrue(filename.with_suffix('.png').is_file())
            self.assertGreater(env.stopped, 0)

    def test_resume_reproduces_next_batch_and_weights(self):
        c = self.settings()
        with tempfile.TemporaryDirectory() as temp:
            dirs=[]
            for label in ('a','b'):
                with patch('watercraft.logging.run_path', return_value=Path(temp)/label):
                    dirs.append(create_run(c))
            torch.manual_seed(12);np.random.seed(12)
            train(c, FakePool(), dirs[0])
            snapshot = torch_load(dirs[0]/'checkpoints/latest.pt')
            c['train']['max_iterations']=2
            train(c, FakePool(), dirs[0], copy.deepcopy(snapshot))
            train(c, FakePool(), dirs[1], copy.deepcopy(snapshot))
            a=torch_load(dirs[0]/'checkpoints/final.pth')
            b=torch_load(dirs[1]/'checkpoints/final.pth')
            for key in a:
                np.testing.assert_array_equal(a[key].numpy(),b[key].numpy())

    def test_ppo_trains_resumes_and_exports_a_deployable_actor(self):
        from watercraft.algorithms import ppo
        from watercraft.contract import NumpyPolicy
        from watercraft.config import validate
        from watercraft.results import plot_run
        c = self.settings()
        c['train']['algorithm'] = 'ppo'
        c['algorithm'] = dict(steps_per_iteration=8, learning_rate=0.0003, gamma=0.99, gae_lambda=0.95,
                              clip_range=0.2, epochs=2, minibatch_size=4, value_coef=0.5, entropy_coef=0.0,
                              max_grad_norm=0.5, initial_log_std=-0.5, reward_scale=0.1, stop_success=2,
                              stop_iterations=3)
        validate(c, 'train')
        # A timeout bootstraps from the critic (7); a reached goal does not; episodes do not leak into each other.
        advantages, returns = ppo.compute_gae(np.array([1, 1, 1, 5], dtype=np.float32), np.array([0, 0, 0, 2], dtype=np.float32),
                                              np.array([0, 0, 7, 0], dtype=np.float32), np.array([0, 0, 1, 1], dtype=bool), 1.0, 1.0)
        np.testing.assert_allclose(advantages, [10, 9, 8, 3])
        np.testing.assert_allclose(returns, [10, 9, 8, 5])
        env = FakePool()
        with tempfile.TemporaryDirectory() as temp:
            with patch('watercraft.logging.run_path', return_value=Path(temp) / 'ppo'):
                directory = create_run(c)
            self.assertEqual(ppo.train(c, env, directory)['iteration'], 1)
            c['train']['max_iterations'] = 2
            result = ppo.train(c, env, directory, torch_load(directory / 'checkpoints/latest.pt'))
            self.assertEqual((result['iteration'], result['env_steps']), (2, 16))
            observation = np.random.RandomState(3).uniform(-1, 1, 6).astype(np.float32)
            for name in ('best', 'final'):
                np.testing.assert_allclose(load(directory / f'checkpoints/{name}.pth', c['policy']).act(observation),
                                           NumpyPolicy(directory / f'checkpoints/{name}.npz').act(observation), atol=1e-6)
            plot_run(directory)
            self.assertTrue((directory / 'graphs/training.png').is_file())
            self.assertGreater(env.stopped, 0)

    def test_sac_trains_resumes_and_exports_a_deployable_actor(self):
        from watercraft.algorithms import sac
        from watercraft.contract import NumpyPolicy
        from watercraft.config import validate
        c = self.settings()
        c['train']['algorithm'] = 'sac'
        c['algorithm'] = dict(learning_rate=0.0003, gamma=0.99, tau=0.005, batch_size=4, buffer_size=100,
                              start_steps=2, updates_per_step=1, critic_hidden=16, initial_alpha=0.2,
                              target_entropy=None, reward_scale=0.1, episodes_per_iteration=2, eval_every=1,
                              eval_episodes=1, eval_seed=1, stop_success=2, stop_iterations=3)
        validate(c, 'train')
        # The tanh-corrected log-prob matches torch's TransformedDistribution.
        actor = sac.Actor(c['policy'])
        torch.manual_seed(0)
        observations = torch.randn(5, 6)
        action, log_prob, _ = actor.sample(observations)
        mean, log_std = actor(observations)
        reference = torch.distributions.TransformedDistribution(
            torch.distributions.Normal(mean, log_std.exp()), [torch.distributions.TanhTransform()])
        np.testing.assert_allclose(log_prob.detach().numpy(),
                                   reference.log_prob(action.clamp(-0.999999, 0.999999)).sum(-1).detach().numpy(),
                                   rtol=1e-3, atol=1e-3)
        env = FakePool()
        with tempfile.TemporaryDirectory() as temp:
            with patch('watercraft.logging.run_path', return_value=Path(temp) / 'sac'):
                directory = create_run(c)
            self.assertEqual(sac.train(c, env, directory)['iteration'], 1)
            c['train']['max_iterations'] = 2
            result = sac.train(c, env, directory, torch_load(directory / 'checkpoints/latest.pt'))
            self.assertEqual((result['iteration'], result['env_steps']), (2, 8))
            self.assertEqual(torch_load(directory / 'checkpoints/latest.pt')['buffer']['count'], 8)
            observation = np.random.RandomState(3).uniform(-1, 1, 6).astype(np.float32)
            for name in ('best', 'final'):
                np.testing.assert_allclose(load(directory / f'checkpoints/{name}.pth', c['policy']).act(observation),
                                           NumpyPolicy(directory / f'checkpoints/{name}.npz').act(observation), atol=1e-6)
            from watercraft.results import plot_run
            plot_run(directory)
            self.assertTrue((directory / 'graphs/training.png').is_file())
            self.assertGreater(env.stopped, 0)

    def test_bad_config_and_paths_fail_before_devices(self):
        from watercraft.config import validate
        c=self.settings();c['algorithm']['batch_size']=0
        with self.assertRaises(ValueError):validate(c,'train')
        with self.assertRaises(ValueError):run_path('../outside')

    def test_best_checkpoint_matches_collected_model(self):
        c=self.settings()
        torch.manual_seed(9)
        expected=copy.deepcopy(build(c['policy']).state_dict())
        with tempfile.TemporaryDirectory() as temp:
            with patch('watercraft.logging.run_path',return_value=Path(temp)/'best'):
                directory=create_run(c)
            torch.manual_seed(9)
            train(c,FakePool(),directory)
            best=torch_load(directory/'checkpoints/best.pth')
            for key in expected:
                np.testing.assert_array_equal(best[key].numpy(),expected[key].numpy())

    def test_json_evaluation_reads_new_and_historical_configs(self):
        import watercraft.evaluate as entrypoint
        c = self.settings()
        c['evaluate'] = resolve('evaluate')['evaluate']
        c['evaluate'].update(checkpoint='best.pth', episodes=1, auto_plot=False)
        for historical in (False, True):
            with self.subTest(historical=historical), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                (directory / 'checkpoints').mkdir()
                torch.save(build(c['policy']).state_dict(), directory / 'checkpoints/best.pth')
                saved = {'hyperparams': {'hidden_size': c['policy']['hidden_size']}, 'position_scale_m': 20} if historical else c
                (directory / 'config.json').write_text(json.dumps(saved))
                with patch.object(entrypoint, 'resolve', return_value=copy.deepcopy(c)), patch.object(entrypoint, 'run_path', return_value=directory), patch.object(entrypoint, 'make', return_value=FakePool()):
                    entrypoint.main()
                metadata = list((directory / 'evaluations/sim').glob('*.json'))
                self.assertEqual(len(metadata), 1)
                self.assertEqual(json.loads(metadata[0].read_text())['status'], 'complete')
                self.assertFalse(list(directory.rglob('*.yaml')))

    def test_geometry_retains_existing_frame_convention(self):
        from watercraft.deploy.vision import rectangle_from_centers
        corners=np.array([[0,0,2],[2,0,2],[2,1,2],[0,1,2]],dtype=float)
        geometry=rectangle_from_centers(corners)
        np.testing.assert_allclose(geometry['center'],[1,.5,2])
        np.testing.assert_allclose(geometry['T_origin_from_camera'] @ geometry['T_camera_from_origin'],np.eye(4))

if __name__ == '__main__':
    unittest.main()
