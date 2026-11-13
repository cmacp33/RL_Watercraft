## 2606 RL-Underactuated Watercraft

### Key locations

| What you need | Location |
| --- | --- |
| Configuration files | `configs/` — all settings use JSON |
| Training setup | `configs/train.json` — choose environment, algorithm, run name, seed and training budget |
| Start training | Run `python train.py` from the repository root |
| Algorithm implementations | `watercraft/algorithms/` — current implementation: `cem.py` (xentropy) |
| Algorithm settings | `configs/algorithms/` — current settings: `cem.json` |
| Training outputs | `runs/<run-name>/` — saved config, checkpoints, CSV metrics and graphs |

To add an algorithm, put its learning code in `watercraft/algorithms/<name>.py`
and its settings in `configs/algorithms/<name>.json`. Connect it to the selection
in `train.py` and the supported-algorithm validation in `watercraft/config.py`;
then choose its name in `configs/train.json`. Adding a file alone does not register
an algorithm. Environment interaction and policy loading should reuse the shared code.

### Training/RL 
Training starts at `train.py`; learning implementations are in
`watercraft/algorithms/`. The older `training_scripts/` filenames remain as
compatibility entry points.

### Gym Environment 
The gym env is defined in:
```
gym_gazebo/envs/pool_gym_env/pool_gym_v0.py
```

### Gazebo Environment
All gazebo stuff (worlds,urdfs,models,launch files,etc) is in:
```
gym_gazebo/envs/ros_ws/src/boat_gazebo/
```

### Hydrodynamics Plugins
The hydrodynamics plugins were cloned from https://github.com/osrf/vrx/tree/gazebo_classic and can be found in:
```
gym_gazebo/env/ros_ws/src/(wave_gazebo_plugins & usv_gazebo_plugins)
```
# 2654 Reinforcement Learning on Underactuated Watercraft

All settings are in `configs/`. Commands take no arguments. CEM is the implemented
training algorithm; PPO/SAC adapters can be added in `watercraft/algorithms/` later.
The existing Gym environment, launch files, models, ROS packages and workspace
structure have been preserved.

## Setup

Use the project's working ROS/Gazebo Python environment for simulation. From this
checkout, install the package in editable mode so tools and older script paths work:

```bash
python -m pip install -e '.[train,results]'
```

For the camera/serial machine:

```bash
python -m pip install -e '.[train,hardware,results]'
```

The `sim` optional group adds Gym if it is missing; use the Gym version already
working on your simulation machine. ROS, Gazebo and Catkin are system dependencies;
Python package installation does not install them. The original installation notes
remain in `INSTALL.md` and the preserved copies under `legacy/`.

## Train, evaluate and view results

1. Edit `configs/train.json`: choose `"environment": "sim"` or `real`, the algorithm,
   unique run name and seed. Algorithm settings are in `configs/algorithms/cem.json`.
   Sim options are in `configs/sim.json`; physical device settings are in `configs/real.json`.
2. Run `python train.py`.
3. Edit `configs/evaluate.json`: choose the environment, run and checkpoint filename,
   for example `best.npz`. Evaluation automatically loads the run's saved policy settings.
4. Run `python evaluate.py`.
5. Edit `configs/results.json` to select runs, then run `python plot.py` or
   `python compare.py`. An empty run list selects existing runs with metrics.

Real trials require `"arm": true` in `configs/real.json` and an operator readiness
acknowledgement before each episode. Collection faults stop motor commands. Learner
updates happen between collections with motors stopped. These paths still require
bench and water validation on the actual devices before relying on them. Camera
reads and firmware watchdog response must be verified on that setup; host code alone
cannot establish the physical stopping bound. Existing firmware behavior is preserved,
including its currently configured transmitter timeout.

Resume training by setting `resume` to `runs/<name>/checkpoints/latest.pt` in
`configs/train.json`, retaining the original environment/model/algorithm settings.
Budgets can be extended. Set `initialize_from` instead to start a new run from
compatible `.pth`, `.pt` or `.npz` actor weights. Neither option overwrites old runs.

## Output locations

```text
runs/<name>/
├── config.json                  # fully resolved training settings
├── checkpoints/
│   ├── best.pth / best.npz      # exact actor used by the best scored collection
│   ├── latest.pt                # learner, optimizer, elite memory and RNG state
│   └── final.pth / final.npz
├── training.csv                 # per-episode metrics
├── iterations.csv               # per-update metrics
├── summary.json
├── tensorboard/                 # when TensorBoard is installed
├── graphs/training.png          # generated after training or by plot.py
├── transitions.csv              # physical collection only
└── evaluations/
    ├── sim/<timestamp>.csv/.json/.png
    └── real/<timestamp>.csv/.json/.png
```

Each evaluation's JSON identifies its checkpoint/hash and settings. Comparisons go
to `runs/comparisons/` by default. Existing historical runs keep their original
files and are readable by plotting/evaluation. New outputs are ignored by Git;
back up run folders separately. `best` is selected using exploratory training
collection, not an independent validation score; use `evaluate.py` for frozen-policy results.

## Gym and ROS stay where they are

```text
gym_gazebo/
├── envs/
│   ├── pool_gym_env/pool_gym_v0.py
│   ├── gazebo_env.py
│   └── ros_ws/
│       ├── src/                 # original ROS packages, launch/world/URDF and models
│       ├── build/               # Catkin generated, when built locally
│       ├── devel/               # Catkin generated
│       ├── install/             # when installed
│       └── log/ or logs/        # when generated
├── spaces/
├── utils/
├── wrappers/
└── tests/
```

No Gym/ROS files were relocated or rewritten. Build/devel/install/log paths remain
ignored; already tracked generated files remain tracked in this change so the
existing workspace is preserved. Ignore rules do not untrack them.
`watercraft/envs/sim.py` configures the existing environment from JSON and stops
only its owned launch process rather than calling the legacy global cleanup routine.

## Other files and tools

`watercraft/` contains shared contract, policy, logging, algorithm and hardware code.
`policy_contract.py`, `run_model.py`, `plot_runs.py`, `compare_runs.py` and
`training_scripts/` retain compatibility imports/entry points using config files.
The alternative tracking implementations remain under `vision_tracking/`, with
settings now loaded from JSON. Their original exact sources are also preserved.
Firmware is under `firmware/transmitter/` and `firmware/receiver/`; board and port
settings remain in each `platformio.ini`. Full original firmware directories,
including local build products, are retained in `legacy/local_firmware/`.

After the editable installation:

```bash
python tools/camera_preview.py
python tools/list_cameras.py
python tools/calibrate_camera.py
python tools/generate_markers.py
python tools/test_sim.py
python tools/test_reset.py
python -m unittest discover -s tests -v
```

Tool settings are in `configs/tools.json`; camera/serial settings are in
`configs/real.json`. Calibration output is backed up before replacement. Existing
calibration files, images and marker print assets are preserved under both their
original paths and the organized `vision_tracking/calibration/` and `assets/markers/` paths.

`legacy/main/` and `legacy/tracy-reorg/` preserve original work from both worktrees.
`legacy/preserved_files.json` records the original file hashes. These copies are
historical references; use the new root entry points for experiments. Source and
settings in the original `/Users/tracysun/Projects/Capstone` worktree are untouched.

## Current repository structure

```text
Capstone-tracy-reorg/
├── train.py                         # config-based training entry point
├── evaluate.py                      # evaluate a saved checkpoint
├── plot.py                          # generate run graphs
├── compare.py                       # compare selected runs
├── configs/
│   ├── train.json                   # environment, algorithm, run, budget, seed, resume
│   ├── evaluate.json                # environment, run, checkpoint and episode count
│   ├── results.json                 # runs and plotting/comparison settings
│   ├── task.json                    # goal, reward and episode settings
│   ├── sim.json                     # existing Gym environment settings
│   ├── real.json                    # camera, serial, calibration and motor settings
│   ├── policy.json                  # observation scales and network dimensions
│   ├── tools.json                   # camera/calibration/diagnostic settings
│   └── algorithms/cem.json          # CEM learning and exploration settings
├── watercraft/
│   ├── config.py                    # read and validate settings
│   ├── contract.py                  # shared observations/actions
│   ├── task.py                      # physical task reward and metrics
│   ├── policy.py                    # model, inference and export
│   ├── logging.py                   # run/checkpoint/CSV handling
│   ├── results.py                   # plotting and comparison functions
│   ├── algorithms/cem.py            # current training algorithm
│   ├── envs/
│   │   ├── sim.py                   # configure the existing Gym environment
│   │   └── real.py                  # physical collection and operator resets
│   └── hardware/
│       ├── vision.py                # ArUco geometry and state estimation
│       └── motors.py                # calibrated serial commands
├── gym_gazebo/                      # preserved existing Gym structure
│   └── envs/
│       ├── pool_gym_env/            # existing pool environment
│       ├── gazebo_env.py
│       └── ros_ws/
│           ├── src/                 # ROS packages, launch, worlds, models and plugins
│           ├── build/               # generated locally by Catkin
│           ├── devel/               # generated locally by Catkin
│           └── install/, log/, logs/ # present when generated locally
├── firmware/
│   ├── transmitter/
│   └── receiver/
├── tools/                           # camera, calibration, marker and sim diagnostics
├── assets/markers/                  # markers and print assets
├── tests/                           # contract and refactor checks
├── vision_tracking/                 # tracking implementations and calibration
│   └── calibration/                 # camera calibration and images
├── training_scripts/                # compatibility entry points
├── saved_weights/                   # preserved historical weights
├── runs/                            # historical and new experiment outputs
├── legacy/                          # preserved original work from both worktrees
├── docs/
└── pyproject.toml                   # package and dependency settings
```

### Train with this structure

Run commands from the `tracy-reorg` worktree. On the simulation machine, first
activate the working ROS/Gazebo Python environment and source its ROS and Catkin
workspace setup as usual. The ROS workspace remains at
`gym_gazebo/envs/ros_ws/`.

```bash
cd /Users/tracysun/Projects/Capstone-tracy-reorg
python -m pip install -e '.[train,results]'
```

On another machine, use that machine's checkout path. Installation is needed once
per Python environment; it does not install ROS or Gazebo.

1. Edit `configs/train.json`. For a new simulation run:

   ```json
   {
     "environment": "sim",
     "algorithm": "cem",
     "run": "sim_cem_01",
     "seed": 42,
     "minutes": 60,
     "max_iterations": null,
     "resume": null,
     "initialize_from": null,
     "auto_plot": true
   }
   ```

   Choose a new `run` name for each experiment. `minutes` is the training time
   budget, checked between batches; `max_iterations` can optionally limit updates.
   CEM is currently the only implemented algorithm.

2. Adjust `configs/algorithms/cem.json` for batch size, learning rate, exploration,
   elite memory and stopping settings. Adjust `configs/sim.json`, `configs/task.json`
   and `configs/policy.json` for the environment, task and policy respectively.

3. Start training, with no command-line options:

   ```bash
   python train.py
   ```

   Outputs go to `runs/sim_cem_01/`: checkpoints in `checkpoints/`, episode metrics
   in `training.csv`, update metrics in `iterations.csv`, TensorBoard logs in
   `tensorboard/` when available, and the graph in `graphs/training.png` when
   `auto_plot` is enabled. The exact configuration is saved as `config.json`.

4. To resume that run, keep its original settings and set this in
   `configs/train.json`, then run `python train.py` again:

   ```json
   {
     "resume": "runs/sim_cem_01/checkpoints/latest.pt",
     "initialize_from": null
   }
   ```

   Extend the time/update budget if needed. To start a separate run from existing
   actor weights, instead use a new run name, leave `"resume": null`, and set
   `initialize_from` to the compatible checkpoint path.

5. To evaluate after training, edit `configs/evaluate.json`:

   ```json
   {
     "environment": "sim",
     "run": "sim_cem_01",
     "checkpoint": "best.npz",
     "episodes": 20,
     "seed": 0,
     "noise": 0.0,
     "goal": null,
     "random_goals": false,
     "randomize": false,
     "auto_plot": true
   }
   ```

   Then run:

   ```bash
   python evaluate.py
   ```

   It loads `runs/sim_cem_01/checkpoints/best.npz` without updating weights and
   writes timestamped results, settings and a graph under
   `runs/sim_cem_01/evaluations/sim/`.

For real training, install the hardware dependencies, set `"environment": "real"` and
a new run name in `configs/train.json`, and configure the camera, calibration,
serial connection, physical goal and motor limits in `configs/real.json`. Physical
actuation requires `"arm": true` there and operator readiness before every episode.
The command remains `python train.py`; results use the same run-folder layout.
