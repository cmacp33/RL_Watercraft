# Proposed file structure

All current work is preserved: source code, alternative implementations, local edits, tests, firmware, calibration files, assets, saved weights and run results. Existing work from both `Capstone` and `Capstone-tracy-reorg` must be retained. Conflicting versions are kept separately, never overwritten. This is a proposal only; no source files have been moved.

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
│   ├── train.yaml                    # sim/real, algorithm, run, seed, resume/initial weights
│   ├── evaluate.yaml                 # sim/real, run, checkpoint, trial cases/count
│   ├── results.yaml                  # runs to plot/compare and output locations
│   ├── task.yaml                     # goals, reward, success and episode duration
│   ├── sim.yaml                      # launch path, stepping, robot parameters, randomization
│   ├── real.yaml                     # camera, calibration, serial, motor settings, limits, arming
│   ├── policy.yaml                   # observation scales, network architecture, inference/export
│   ├── tools.yaml                    # calibration, preview, marker and diagnostic settings
│   └── algorithms/
│       ├── cem.yaml                  # existing training algorithm settings
│       ├── ppo.yaml                  # add when implemented
│       └── sac.yaml                  # add when implemented
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
│   │   ├── sim.py                    # Gazebo lifecycle, state, step and reset
│   │   └── real.py                   # timed camera/motor steps and operator reset
│   └── hardware/
│       ├── __init__.py
│       ├── vision.py                 # current marker geometry methods and state estimation
│       └── motors.py                 # motor calibration, serial commands and stop handling
│
├── simulation/
│   └── ros_ws/
│       └── src/                      # preserve all existing ROS source and model assets
│           ├── boat_gazebo/          # launch files, worlds, URDF, models and diagnostic nodes
│           ├── usv_gazebo_plugins/   # existing source, headers, tests and notices
│           └── wave_gazebo_plugins/  # existing source, headers, tests and notices
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
├── calibration/
│   ├── camera_calibration.npz         # preserve current calibration
│   └── images/                       # preserve all current calibration images
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
        ├── config.yaml               # resolved settings used for this run
        ├── checkpoints/              # weights, training state, optional NumPy export
        ├── training.csv
        └── evaluations/
            ├── sim/
            └── real/
```

Existing run folders retain their original files and layouts. The illustrated run layout applies to new runs. Optional TensorBoard logs and real-training transition recordings stay inside their run. Generated ROS/PlatformIO output and installed environments remain available locally and ignored by Git; they are not source to discard. Original filenames and paths are retained beneath the appropriate `legacy/` worktree folder for preserved Python implementations. The existing ROS package names, model asset paths and firmware project contents stay intact when moved. Firmware pins, timing and motor parameters belong in firmware configuration headers, alongside PlatformIO build settings, rather than Python scripts.

# How to call each file

Run these commands from the repository root. Edit the indicated configuration files first; commands take no arguments.

| Purpose | Configuration to edit | Command |
| --- | --- | --- |
| Train in simulation | `configs/train.yaml`: environment `sim`, algorithm and run; `sim.yaml`, `task.yaml`, `policy.yaml`, selected algorithm YAML | `python train.py` |
| Train on the real boat | `configs/train.yaml`: environment `real`, algorithm and run; `real.yaml`, `task.yaml`, `policy.yaml`, selected algorithm YAML | `python train.py` |
| Evaluate in simulation | `configs/evaluate.yaml`: environment `sim`, run and checkpoint; `sim.yaml`, `task.yaml`, `policy.yaml` | `python evaluate.py` |
| Evaluate on the real boat | `configs/evaluate.yaml`: environment `real`, run and checkpoint; `real.yaml`, `task.yaml`, `policy.yaml` | `python evaluate.py` |
| Plot results | `configs/results.yaml` | `python plot.py` |
| Compare runs | `configs/results.yaml` | `python compare.py` |
| Calibrate camera | `configs/tools.yaml` and `configs/real.yaml` | `python tools/calibrate_camera.py` |
| Preview camera/tracking | `configs/tools.yaml` and `configs/real.yaml` | `python tools/camera_preview.py` |
| List cameras | `configs/tools.yaml` | `python tools/list_cameras.py` |
| Generate markers | `configs/tools.yaml` | `python tools/generate_markers.py` |
| Run simulator diagnostic | `configs/tools.yaml` and `configs/sim.yaml` | `python tools/test_sim.py` |
| Check simulator reset | `configs/tools.yaml` and `configs/sim.yaml` | `python tools/test_reset.py` |
| Run tests | Test fixtures/configuration inside `tests/` | `python -m unittest discover -s tests` |
| Build transmitter | `firmware/transmitter/platformio.ini` and its configuration header | `cd firmware/transmitter` then `pio run` |
| Build receiver | `firmware/receiver/platformio.ini` and its configuration header | `cd firmware/receiver` then `pio run` |

Files under `watercraft/` are imported by these scripts, not run separately. `train.py` selects an algorithm from `configs/train.yaml`; changing algorithms does not require another training entry point. Both environments share the task and policy contract. Real training and evaluation share the same vision/motor code. `evaluate.py` loads the selected checkpoint without learning. Resume, initialization, exploration, physical reset/readiness and arming settings are configured in the YAML files. Physical readiness acknowledgements remain runtime interactions, not command-line settings.

All listed commands and new paths are proposed interfaces pending approval.
