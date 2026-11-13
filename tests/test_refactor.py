"""Integration checks without opening ROS, cameras, serial ports or motors."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch
from watercraft.config import resolve, ROOT, run_path
from watercraft.policy import build, export, load, TorchPolicy
from watercraft.logging import create_run, CsvLog
from watercraft.algorithms.cem import train, episode_score, Episode
from evaluate import evaluate, EVAL_FIELDS


class FakePool:
    is_real = True  # fake stop/close, without importing ROS
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
    def stop(self):
        self.stopped += 1
    def close(self):
        self.stop()


class RefactorTests(unittest.TestCase):
    def settings(self):
        c = resolve('train')
        c['algorithm'].update(batch_size=2, train_steps_per_iteration=1, stop_success=2)
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
            resumed = torch.load(checkpoint, weights_only=False)
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
            snapshot = torch.load(dirs[0]/'checkpoints/latest.pt', weights_only=False)
            c['train']['max_iterations']=2
            train(c, FakePool(), dirs[0], copy.deepcopy(snapshot))
            train(c, FakePool(), dirs[1], copy.deepcopy(snapshot))
            a=torch.load(dirs[0]/'checkpoints/final.pth', weights_only=True)
            b=torch.load(dirs[1]/'checkpoints/final.pth', weights_only=True)
            for key in a:
                torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)

    def test_bad_config_and_paths_fail_before_devices(self):
        from watercraft.config import validate
        c=self.settings();c['algorithm']['batch_size']=0
        with self.assertRaises(ValueError):validate(c,'train')
        with self.assertRaises(ValueError):run_path('../outside')
        from watercraft.envs.real import RealEnv
        with self.assertRaisesRegex(ValueError,'arm'):
            RealEnv(resolve('train'),'train',42)

    def test_best_checkpoint_matches_collected_model(self):
        c=self.settings()
        torch.manual_seed(9)
        expected=copy.deepcopy(build(c['policy']).state_dict())
        with tempfile.TemporaryDirectory() as temp:
            with patch('watercraft.logging.run_path',return_value=Path(temp)/'best'):
                directory=create_run(c)
            torch.manual_seed(9)
            train(c,FakePool(),directory)
            best=torch.load(directory/'checkpoints/best.pth',weights_only=True)
            for key in expected:
                torch.testing.assert_close(best[key],expected[key],rtol=0,atol=0)

    def test_json_evaluation_reads_new_and_historical_configs(self):
        import evaluate as entrypoint
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

    def test_preserved_files_have_original_hashes(self):
        records=json.loads((ROOT/'legacy/preserved_files.json').read_text())
        for record in records:
            f=ROOT/'legacy'/record['worktree']/record['path']
            self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(),record['sha256'],str(f))

    def test_geometry_retains_existing_frame_convention(self):
        from watercraft.hardware.vision import rectangle_from_centers
        corners=np.array([[0,0,2],[2,0,2],[2,1,2],[0,1,2]],dtype=float)
        geometry=rectangle_from_centers(corners)
        np.testing.assert_allclose(geometry['center'],[1,.5,2])
        np.testing.assert_allclose(geometry['T_origin_from_camera'] @ geometry['T_camera_from_origin'],np.eye(4))

    def test_physical_episode_and_fault_stop(self):
        from watercraft.envs.real import RealEnv
        class Motors:
            def __init__(self):self.stops=0
            def act(self,action):return (1,1)
            def stop(self):self.stops+=1
            def close(self):self.stop()
        class Camera:
            def reset(self):pass
            def read(self):return dict(x=.25,y=.25,yaw=0.,vx=0.,vy=0.,yaw_rate=0.,timestamp=time.monotonic(),inside=True)
            def close(self):pass
        c=self.settings();c['real']['arm']=True;c['real']['control_period_s']=.001
        m=Motors();camera=Camera()
        e=RealEnv(c,'train',42,camera,m,lambda:None)
        try:
            e.reset()
            obs,reward,done,info=e.step([0,0])
            self.assertTrue(done)
            self.assertTrue(info['reached'])
            with patch.object(camera,'read',side_effect=RuntimeError('marker lost')):
                with self.assertRaisesRegex(RuntimeError,'marker lost'):e.step([.1,.1])
            self.assertGreaterEqual(m.stops,3)
        finally:e.close()

if __name__ == '__main__':
    unittest.main()
