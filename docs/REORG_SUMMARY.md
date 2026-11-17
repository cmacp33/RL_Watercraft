# Reorganisation summary

Branch `tracy-reorg`, 2026-11-12. The layout and its rules are in
[REFACTORING_PLAN.md](REFACTORING_PLAN.md); commands are in the README.

## What changed

- **Python code is under `src/`.** `watercraft/` and `gym_gazebo/` moved there.
  `gym_gazebo/` was moved as one unit; its 133 tracked files are byte-identical.
- **One entry point per task.** `python -m watercraft.train`,
  `python -m watercraft.evaluate`, `python -m watercraft.deploy`. The root scripts
  and compatibility wrappers are gone.
- **Real boat is deployment only.** `vision_tracking/src/track_aruco_3d.py` was
  split into `src/watercraft/deploy/` (`__main__.py`, `vision.py`, `motors.py`)
  with its code unchanged. Real-boat training was removed.
- **Four config files instead of nine.** `configs/train.json`,
  `configs/evaluate.json`, `configs/deploy.json`, `vision_tracking/config.json`.
- **Git tracks source only.** About 3,200 files removed from tracking: the
  `vision_tracking` virtualenv, Catkin `build/` and `devel/`, `legacy/`,
  `.DS_Store` files. Tracked files went from 3,505 to 309.
- **`vision_tracking/`** now holds calibration data, markers and the camera
  scripts.

## Code changes beyond moving files

- `config.py` reads sections from the merged config files.
- `train.algorithm` in `train.json` is the only place an algorithm is selected.
- Deployment reads the policy scales from the `config.json` saved beside the
  weights.
- `policy.py` loads checkpoints on torch older than 1.13 (see below).

## Test results

Run on the simulation machine (Python 3.8, torch 1.5.1, ROS Noetic), same config
and seed before and after, two training iterations, headless Gazebo.

| Check | Before | After |
| --- | --- | --- |
| Sim training | Crashed after the first batch of 8 episodes | Completed both iterations |
| Iteration 0, mean final distance | 15.24 m | 15.29 m |
| Unit tests | 16 run, 6 errors | 14 run, all pass |

- The eight episodes of iteration 0 match the baseline episode by episode to
  within about 0.2 m.
- The crash before the reorganisation was `torch.load(..., weights_only=False)`,
  which torch 1.5.1 does not accept. It is fixed.
- The two tests no longer present covered `legacy/` file hashes and real-boat
  training.
- Evaluation, plotting, comparison, checkpoint export and the sim diagnostic all
  ran successfully on the new layout.
- Editable install and the six `watercraft-*` commands were confirmed with a
  current pip.

## Not tested

- Deployment on hardware. The controller was exercised in dry-run mode only; the
  camera loop and serial port were not. Do a dry run (`"arm": false`) on the boat
  laptop before driving the motors.

## After pulling

- Rebuild the ROS workspace: `cd src/gym_gazebo/envs/ros_ws && catkin_make`, and
  source `src/gym_gazebo/envs/ros_ws/devel/setup.bash` (update any shell profile
  that sources the old path).
- Reinstall with `pip install -e .` (needs pip 21.3 or newer), or set
  `PYTHONPATH` to the `src` folder.
