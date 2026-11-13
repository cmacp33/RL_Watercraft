# Proposed file structure

All current work is preserved: source code, alternative implementations, local edits, tests, firmware, calibration files, assets, saved weights and run results. Existing work from both `Capstone` and `Capstone-tracy-reorg` must be retained. Conflicting versions are kept separately, never overwritten. The surrounding structure is implemented; the existing Gym/ROS workspace remains in place. See README.md for current commands and verification limits.

All settings are in config files. No `argparse`, command-line options or settings hardcoded in Python entry points.

```text
Capstone/
├── README.md
├── INSTALL.md
├── pyproject.toml                    # Python package/dependencies
├── .gitignore
│
├── train.py                          # train in sim or real
├── evaluate.py                       # evaluate in sim or real
├── plot.py                           # plot training/evaluation results
├── compare.py                        # compare selected runs
│
├── configs/
│   ├── train.json                    # sim/real, algorithm, run, seed, resume/initial weights
│   ├── evaluate.json                 # sim/real, run, checkpoint, trial cases/count
│   ├── results.json                  # runs to plot/compare and output locations
│   ├── task.json                     # goals, reward, success and episode duration
│   ├── sim.json                      # launch path, stepping, robot parameters, randomization
│   ├── real.json                     # camera, calibration, serial, motor settings, limits, arming
│   ├── policy.json                   # observation scales, network architecture, inference/export
│   ├── tools.json                    # calibration, preview, marker and diagnostic settings
│   └── algorithms/
│       ├── cem.json                  # existing training algorithm settings
│       ├── ppo.json                  # add when implemented
│       └── sac.json                  # add when implemented
│
├── watercraft/                       # shared Python code, imported by entry points
│   ├── __init__.py
│   ├── config.py                     # read config files
│   ├── contract.py                   # shared observations and normalized actions
│   ├── task.py                       # shared reward, success, failure and timeout
│   ├── policy.py                     # model definitions, loading, inference and export
│   ├── logging.py                    # run settings, checkpoints and CSV writing
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── cem.py                    # current CEM training logic
│   │   ├── ppo.py                    # add when implemented
│   │   └── sac.py                    # add when implemented
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── sim.py                    # configure existing Gym environment
│   │   └── real.py                   # timed camera/motor steps and operator reset
│   └── hardware/
│       ├── __init__.py
│       ├── vision.py                 # current marker geometry methods and state estimation
│       └── motors.py                 # motor calibration, serial commands and stop handling
│
├── gym_gazebo/                       # preserved existing Gym structure
│   ├── envs/
│   │   ├── pool_gym_env/             # existing environment implementation
│   │   ├── gazebo_env.py
│   │   └── ros_ws/
│   │       ├── src/                  # existing ROS packages and assets
│   │       ├── build/                # local generated ROS output
│   │       ├── devel/                # local generated ROS output
│   │       ├── install/              # when generated
│   │       └── log/ or logs/         # when generated
│   ├── spaces/
│   ├── wrappers/
│   ├── utils/
│   └── tests/
│
├── firmware/
│   ├── transmitter/                  # preserve complete existing PlatformIO project
│   │   └── platformio.ini            # board, build, upload and monitor settings
│   └── receiver/                     # preserve complete existing PlatformIO project
│       └── platformio.ini
│
├── tools/
│   ├── calibrate_camera.py
│   ├── camera_preview.py
│   ├── list_cameras.py
│   ├── generate_markers.py
│   ├── test_sim.py                    # existing simulator action diagnostic
│   └── test_reset.py                  # existing reset diagnostic
│
├── vision_tracking/
│   └── calibration/
│       ├── camera_calibration.npz     # preserve current calibration
│       └── images/                   # preserve all current calibration images
├── assets/
│   └── markers/                      # preserve all marker images and print documents
├── tests/                            # preserve existing tests; shared behavior tests
├── docs/
│   ├── REFACTORING_PLAN.md
│   └── previous_plans/               # preserve earlier plans and review notes
│
├── legacy/                           # preserve original Python implementations and helpers
│   ├── main/                         # originals/local versions from the main worktree
│   └── tracy-reorg/                  # differing originals/local versions from this worktree
│
├── saved_weights/                    # preserve all current weights unchanged
└── runs/
    └── <run-name>/
        ├── config.json               # resolved settings used for this run
        ├── checkpoints/              # weights, training state, optional NumPy export
        ├── training.csv
        └── evaluations/
            ├── sim/
            └── real/
```

Existing run folders retain their original files and layouts. The illustrated run layout applies to new runs. Optional TensorBoard logs and real-training transition recordings stay inside their run. Generated ROS/PlatformIO output and installed environments remain available locally and ignored by Git; they are not source to discard. Original filenames and paths are retained beneath the appropriate `legacy/` worktree folder for preserved Python implementations. The entire existing Gym/ROS tree stays in place. Firmware source versions and complete original projects are preserved. Firmware pins, timing and motor parameters belong in firmware configuration headers, alongside PlatformIO build settings, rather than Python scripts.

# How to call each file

Run these commands from the repository root. Edit the indicated configuration files first; commands take no arguments.

| Purpose | Configuration to edit | Command |
| --- | --- | --- |
| Train in simulation | `configs/train.json`: environment `sim`, algorithm and run; `sim.json`, `task.json`, `policy.json`, selected algorithm JSON | `python train.py` |
| Train on the real boat | `configs/train.json`: environment `real`, algorithm and run; `real.json`, `task.json`, `policy.json`, selected algorithm JSON | `python train.py` |
| Evaluate in simulation | `configs/evaluate.json`: environment `sim`, run and checkpoint; `sim.json`, `task.json`, `policy.json` | `python evaluate.py` |
| Evaluate on the real boat | `configs/evaluate.json`: environment `real`, run and checkpoint; `real.json`, `task.json`, `policy.json` | `python evaluate.py` |
| Plot results | `configs/results.json` | `python plot.py` |
| Compare runs | `configs/results.json` | `python compare.py` |
| Calibrate camera | `configs/tools.json` and `configs/real.json` | `python tools/calibrate_camera.py` |
| Preview camera/tracking | `configs/tools.json` and `configs/real.json` | `python tools/camera_preview.py` |
| List cameras | `configs/tools.json` | `python tools/list_cameras.py` |
| Generate markers | `configs/tools.json` | `python tools/generate_markers.py` |
| Run simulator diagnostic | `configs/tools.json` and `configs/sim.json` | `python tools/test_sim.py` |
| Check simulator reset | `configs/tools.json` and `configs/sim.json` | `python tools/test_reset.py` |
| Run tests | Test fixtures/configuration inside `tests/` | `python -m unittest discover -s tests` |
| Build transmitter | `firmware/transmitter/platformio.ini` and its configuration header | `cd firmware/transmitter` then `pio run` |
| Build receiver | `firmware/receiver/platformio.ini` and its configuration header | `cd firmware/receiver` then `pio run` |

Files under `watercraft/` are imported by these scripts, not run separately. `train.py` selects an algorithm from `configs/train.json`; changing algorithms does not require another training entry point. Both environments share the task and policy contract. Real training and evaluation share the same vision/motor code. `evaluate.py` loads the selected checkpoint without learning. Resume, initialization, exploration, physical reset/readiness and arming settings are configured in the JSON files. Physical readiness acknowledgements remain runtime interactions, not command-line settings.

Implemented entry points read configuration files. CEM is currently implemented; PPO/SAC remain future adapters. Verification details are in README.md.
