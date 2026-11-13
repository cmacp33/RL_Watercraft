# Repository structure and rules

This records the structure the repository was reorganised into and the rules
behind it. The README has the day-to-day commands.

## Rules

- The top level matches what you do: train and evaluate (`src/watercraft/`),
  deploy (`src/watercraft/deploy/`), camera work (`vision_tracking/`), firmware
  (`firmware/`), settings (`configs/`).
- All Python packages live under `src/`. No Python files at the repository root.
- `gym_gazebo/` is moved to `src/` as one unit and is otherwise not edited,
  rearranged or trimmed. The ROS workspace stays inside it.
- One config file per command: `configs/train.json`, `configs/evaluate.json`,
  `configs/deploy.json`, plus `vision_tracking/config.json` for the camera scripts.
- Policy settings live in `train.json`. Evaluation and deployment read them from
  the `config.json` saved beside the checkpoint, not from `configs/`.
- One way to do each thing: no compatibility wrappers, except
  `src/policy_contract.py`, which `gym_gazebo` imports.
- The real boat is deployment only. There is no real-boat training or evaluation
  environment.
- Nothing generated is tracked: virtual environments, Catkin `build/` and
  `devel/`, `egg-info`, `.DS_Store`, new runs.
- No copies of old versions in the tree; git history keeps them.
- One virtual environment per machine at the repository root, with the extras in
  `pyproject.toml` selecting what is installed.

## Structure

```text
RL-Watercraft/
├── README.md  pyproject.toml  .gitignore
├── configs/            train.json  evaluate.json  deploy.json
├── src/
│   ├── watercraft/
│   │   ├── train.py  evaluate.py  results.py
│   │   ├── deploy/     __main__.py  vision.py  motors.py
│   │   ├── algorithms/ cem.py
│   │   └── sim_env.py  policy.py  contract.py  config.py  logging.py
│   ├── gym_gazebo/     unchanged, including envs/ros_ws/
│   └── policy_contract.py
├── firmware/           transmitter/  receiver/
├── vision_tracking/    config.json  calibration/  markers/  scripts/
├── tests/
└── docs/
```

## Where the earlier files went

| Before | Now |
| --- | --- |
| `train.py`, `evaluate.py` | `src/watercraft/train.py`, `src/watercraft/evaluate.py` |
| `plot.py`, `compare.py` | `plot_main` and `compare_main` in `src/watercraft/results.py` |
| `training_scripts/export_policy.py` | `export_main` in `src/watercraft/policy.py` |
| `watercraft/envs/sim.py` | `src/watercraft/sim_env.py` |
| `vision_tracking/src/track_aruco_3d.py` | split into `src/watercraft/deploy/` |
| `watercraft/hardware/vision.py` | `src/watercraft/deploy/vision.py` (geometry only) |
| `tools/` camera scripts | `vision_tracking/scripts/` |
| `tools/test_sim.py`, `tools/test_reset.py` | `tests/sim_diagnostic.py`, `tests/reset_diagnostic.py` |
| `configs/policy.json`, `task.json`, `sim.json`, `algorithms/cem.json` | sections of `configs/train.json` |
| `configs/results.json` | `results` section of `configs/evaluate.json` |
| `configs/real.json` | `configs/deploy.json` |
| `configs/tools.json` | `vision_tracking/config.json` |
| `vision_tracking/markers/Calibration_images/` | `vision_tracking/calibration/images/` |

Removed: `legacy/`, the compatibility wrappers (`run_model.py`, `plot_runs.py`,
`compare_runs.py`, `setup.py`, the rest of `training_scripts/`), real-boat
training (`watercraft/envs/real.py`, `watercraft/task.py`, the `Motors` and
`CameraState` classes), the older trackers `track_aruco_homo.py` and
`track_aruco_v3.py`, and `saved_weights/`.
