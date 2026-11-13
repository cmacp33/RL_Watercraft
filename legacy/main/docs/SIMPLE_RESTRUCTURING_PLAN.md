# Simple restructuring plan

Updated: 2026-11-12. This is the preferred plan and supersedes the larger proposed organization in `REFACTORING_PLAN.md`. It is a plan only; the repository has not been moved.

## What we need

One repository that can train in simulation, evaluate in simulation, train on the real boat, and evaluate on the real boat using different algorithms. Keep shared behavior in a few ordinary Python modules. Add folders only where they group several related files.

Use the agreed `runs/<run-name>/` layout for all experiment outputs. Select a checkpoint explicitly. Its configuration and evaluation results stay with the run. No separate policy catalog, release directory, qualification system, or model-card requirement.

## Proposed structure

```text
Capstone/
├── README.md
├── pyproject.toml
├── .gitignore
├── train.py                       # one training entry point: sim or real
├── evaluate.py                    # one evaluation entry point: sim or real
├── plot.py                        # plot/compare training and evaluation results
├── configs/
│   ├── sim.yaml                   # simulator, robot, task and control settings
│   ├── real.yaml                  # camera, serial, calibration, task and limits
│   ├── cem.yaml                   # CEM training parameters
│   ├── ppo.yaml                   # add when PPO is implemented
│   └── sac.yaml                   # add when SAC is implemented
├── watercraft/
│   ├── __init__.py
│   ├── contract.py                # observation/action conventions and conversions
│   ├── task.py                    # reward, success, failure and episode horizon
│   ├── policy.py                  # network definitions, loading, inference, export
│   ├── logging.py                 # config snapshot, CSVs and optional TensorBoard
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── sim.py                 # Gazebo startup, reset, stepping and cleanup
│   │   └── real.py                # camera state, timed actions, operator reset
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── cem.py                 # existing elite-based training
│   │   ├── ppo.py                 # add when needed; library adapter
│   │   └── sac.py                 # add when needed; library adapter
│   └── hardware/
│       ├── __init__.py
│       ├── vision.py              # marker geometry, calibration loading, state estimation
│       └── motors.py              # serial commands, motor calibration, stop handling
├── simulation/
│   └── ros_ws/
│       └── src/
│           ├── boat_gazebo/       # launch, world, robot, meshes and textures
│           ├── usv_gazebo_plugins/
│           └── wave_gazebo_plugins/
├── firmware/
│   ├── transmitter/              # existing PlatformIO project
│   └── receiver/                 # existing PlatformIO project
├── calibration/
│   ├── camera.npz                # current small calibration file
│   └── README.md                 # camera/resolution, marker dimensions and calibration notes
├── tools/
│   ├── calibrate_camera.py
│   ├── camera_preview.py
│   └── generate_markers.py
├── assets/
│   └── markers/                  # marker images and print document
├── tests/                        # contract, environment, learning and hardware-replay tests
├── docs/
│   └── SIMPLE_RESTRUCTURING_PLAN.md
└── runs/                         # ignored experiment outputs
    └── <run-name>/
        ├── config.yaml           # exact settings used, algorithm, backend, seed, source revision
        ├── checkpoints/          # weights, resumable training state and optional .npz exports
        ├── training.csv
        └── evaluations/
            ├── sim/
            └── real/
```

The package is intentionally flat: `watercraft/` is beside the scripts, with no extra `src/` layer. Install it in editable mode for development; the root scripts import the installed package without editing `sys.path`. `pyproject.toml` defines dependencies, with optional training, simulation and hardware groups so the camera machine does not need the full simulation stack.

The algorithm filenames show the intended extension points; do not create empty PPO/SAC implementations now. Keep the existing ROS plugin sources together and preserve their licenses. There is no need to move them into a new third-party management system during this refactor. Firmware can keep duplicated protocol definitions initially; introduce one shared header only when changing the protocol.

## How the four workflows work

| Command | Environment | Learning | Results |
| --- | --- | --- | --- |
| `train.py --env sim` | Gazebo | Selected algorithm updates its model | Run config, checkpoints, training CSV |
| `evaluate.py --env sim` | Gazebo | Disabled | Selected run's `evaluations/sim/` |
| `train.py --env real` | Camera and motors | Selected algorithm learns from physical trials | Run config, checkpoints, training CSV and collected transitions |
| `evaluate.py --env real` | Camera and motors | Disabled | Selected run's `evaluations/real/` |

`train.py` selects the environment and algorithm, loads settings and starts the algorithm's learning loop. `evaluate.py` loads a particular saved policy and runs it through the selected environment. It never imports a training script to obtain a network definition, updates weights, or fits normalization statistics.

Example target commands, not yet implemented:

```bash
python train.py --env sim --algorithm cem --run sim_cem_01 --seed 42
python train.py --env sim --algorithm sac --run sim_sac_01 --seed 42
python evaluate.py --env sim --run sim_cem_01 --checkpoint best.npz --episodes 20
python train.py --env real --algorithm cem --run real_cem_01 --seed 42 --arm
python train.py --env real --algorithm sac --run real_sac_01 --initialize-from runs/sim_sac_01/checkpoints/best.pt --arm
python evaluate.py --env real --run real_sac_01 --checkpoint best.npz --episodes 10 --arm
python train.py --resume runs/real_sac_01/checkpoints/latest.pt
python plot.py runs/sim_cem_01 runs/sim_sac_01
```

Commands load `configs/<env>.yaml` and, for training, `configs/<algorithm>.yaml`; `--env-config` and `--algorithm-config` can select alternative files. Save the fully combined settings in the run's `config.yaml`. Reject contradictory settings on resume. Creating a run with an existing name must fail rather than overwrite data.

Checkpoint names are relative to the selected run's `checkpoints/`. `--initialize-from` starts a new run from compatible actor weights; `--resume` restores full training state. Cross-algorithm initialization is allowed only if an implemented adapter supports it. Real commands start disarmed unless `--arm` is supplied, and resuming never silently rearms. Final CLI details can be adjusted during implementation while preserving these meanings.

## Keep these responsibilities separate

### Shared task and policy contract

`contract.py` owns the current six-value body-relative observation and normalized physical left/right action. Preserve those values during file moves so existing weights retain their meaning. Keep motor gains, channel swaps and serial conversion in `hardware/motors.py`.

`task.py` owns reward, success conditions, time limits and metrics. Both environments call it. Algorithms must not implement separate definitions of goal success. Store raw metric state separately from clipped policy observations so distance and workspace checks remain accurate. Put goal tolerance and any stop-speed/dwell requirement in configuration.

`policy.py` owns model definitions and inference independently of learning. Keep the current NumPy export for the current MLP. Add export support when a new architecture requires it; unsupported architectures must produce an explicit error. Evaluate the exported actor when it is the one used on the boat. Do not assume PPO or SAC fits the existing four-array NumPy format.

### Two environment modules

Both environments expose the same Gymnasium reset/step results and observation/action shapes, including separate `terminated` and `truncated` values. This is the boundary that lets algorithms use either environment. The existing real-environment stubs are not working real training implementations and must be replaced.

`sim.py` owns ROS/Gazebo processes, fresh state, physics stepping and automated resets. Stop thrust before resetting and on close; close only the processes started by this environment. Keep the existing launch/world/robot files under the source-only ROS workspace.

`real.py` composes `vision.py`, `motors.py`, `contract.py` and `task.py`. It runs actions at the configured real-time interval and waits for fresh measurements. Its reset asks the operator to reposition the boat, verifies a valid starting pose and low motion, and waits for readiness before starting. Hardware does not pause or teleport like simulation.

Real training and evaluation use this same control path. The camera overlay is optional display code, not another controller implementation. Stop on lost/stale tracking, invalid actions, workspace exit, operator stop and process exit. Treat faults as recorded faults; do not fabricate missing next observations for training.

### Algorithms own their own learning

Keep one module per implemented algorithm. Each module owns its optimizer, buffers, exploration and update schedule; reuse the environment, task and logger. Keep existing CEM behavior first, then add a maintained PPO/SAC implementation through a small adapter.

CEM needs elite episodes and memory; PPO needs its on-policy rollouts and behavior probabilities; SAC needs replay and critic state. They should not be forced into one CEM-shaped training loop. Save enough state to resume each implemented algorithm, including RNG and optimizer state where applicable.

For initial real training, collect an episode or batch, stop/disarm the boat, then update the learner. Keep one actor snapshot throughout an episode and, where required, the entire on-policy rollout. Install updated weights at a collection boundary. Do not run a long optimization step while retaining the previous motor command. Add asynchronous collection only if later experiments require it.

## Run files: the only experiment structure

```text
runs/sim_cem_01/
├── config.yaml
├── checkpoints/
│   ├── best.pt
│   ├── latest.pt
│   ├── final.pt
│   └── best.npz                   # when this export format is supported
├── training.csv
└── evaluations/
    ├── sim/
    │   ├── 20261112_140000.csv
    │   └── 20261112_140000.yaml
    └── real/
        ├── 20261113_110000.csv
        └── 20261113_110000.yaml
```

Use the library's native checkpoint extension if needed; do not rename files just to force `.pt`. A checkpoint can include weights and learner state together. “Best” must refer to the exact model that was evaluated; it must not be selected using scores collected before different weights were saved. Record the selection rule in the run config. Evaluation records identify the checkpoint by content hash because a file called `best.pt` may later change.

`config.yaml` contains the resolved environment/task/algorithm settings, seed, source revision and whether local edits existed. If edits existed, save the source diff as `source.patch` in the same run when reproducibility is needed. New fine-tuning runs record the parent checkpoint path/hash. Never rewrite historical run configuration to match newer code.

`training.csv` holds common episode metrics: episode, environment steps, elapsed time, reward, success, start/final/minimum distance and end reason. Add algorithm metrics as appropriate; loss values are not comparable across algorithms. Compare learning against environment interactions and time rather than iteration number alone.

Each evaluation CSV holds per-trial metrics. Its small companion YAML records the exact checkpoint/hash, environment settings, explicit sim cases or real trial protocol, calibration reference, firmware revision when relevant, and evaluation seed. These are evaluation settings inside the run, not a separate release system. Never overwrite a previous evaluation.

For real training, add `transitions.csv` in the same run when collecting data: episode/step, actor version, current/next observation, requested action, sent motor command, reward, termination/truncation, timestamps and intervention/fault reason. Actual receiver-applied command is unknown without telemetry. Preserve complete transitions; omit incomplete fault transitions from learning. Additional large recordings can go in that run only when needed. Optional TensorBoard files also stay in the run.

Evaluate fixed policies on separate evaluation cases and repeated trials. Use the same task and action limits when comparing algorithms. Record physical start deviations, failed trials and interventions instead of dropping them. Do not claim noisy camera measurements are ground truth. A run trained in sim may have both sim and real evaluation folders; real fine-tuning creates a new run.

Back up runs to the team's chosen shared storage. Keeping them ignored by Git must not mean keeping the only copy on one laptop. Small calibration files and marker assets can stay in Git; large calibration photo collections need not.

## Where existing files go

| Current content | Simple destination |
| --- | --- |
| `policy_contract.py` | `watercraft/contract.py`; inference to `policy.py`, device calibration to `hardware/motors.py` |
| `training_scripts/gazebo_pool_v0_xentropy.py` | CEM logic to `algorithms/cem.py`, model to `policy.py`, arguments to root `train.py` |
| `training_scripts/run_logging.py` | `watercraft/logging.py` |
| `training_scripts/export_policy.py` | Export function in `policy.py`; expose export through training/checkpoint handling |
| `run_model.py` | Root `evaluate.py`, shared policy loading |
| `plot_runs.py`, `compare_runs.py` | Root `plot.py`, retaining support for old CSVs |
| `pool_gym_v0.py`, `gazebo_env.py` | `envs/sim.py`; observation/reward/termination extracted to `contract.py` and `task.py` |
| `real_env.py`, `real_env_ros2.py` | Replace unused stubs with `envs/real.py` |
| `gym_gazebo/envs/ros_ws/src/` | `simulation/ros_ws/src/`, preserving package/model names |
| `track_aruco_3d.py` | Geometry/estimation to `hardware/vision.py`, motor transmission to `hardware/motors.py`, episode control to `envs/real.py` |
| `track_aruco_v3.py`, `track_aruco_homo.py` | Compare geometry against recorded/known poses before choosing the default; retain useful alternatives as functions in `vision.py`, otherwise retire after validation |
| Calibration, camera and marker scripts | `tools/`; shared functions in `vision.py`; calibration file in `calibration/`; marker prints in `assets/markers/` |
| `ESP32_Reciever/`, `ESP32_transmitter/` | `firmware/receiver/`, `firmware/transmitter/` |
| Existing run directories | Preserve existing data/layout; new runs use the new layout; avoid rewriting old results |
| `saved_weights/pool_policy.pth` | Preserve as a legacy checkpoint with its known settings before retiring the ambiguous global copy |
| Contract/reset tests | `tests/`; keep manual simulator diagnostics explicit |
| Copied Gym spaces/wrappers/utils | Check active references, then remove unused copies; use maintained equivalents where required |
| `qq`, temporary runner files, generated editor settings | Remove only after confirming no useful source/data; `qq` is a Git-log dump |
| README/INSTALL/setup.py | Accurate README instructions and `pyproject.toml` |

Keep existing entry points temporarily as thin wrappers if teammates still use them. Fix paths and imports in the same move; avoid maintaining duplicate implementations. Do not change simulator physics while moving source files.

## Implementation order

1. **Clean Git tracking.** Preserve current edits and run data. Untrack generated ROS output, virtual environments, caches, egg-info and PlatformIO products using targeted `git rm --cached`; leave local files intact. Update ignore rules for the proposed workspace path and `runs/`. Ignore rules alone do not untrack existing files.
2. **Extract shared code.** Create the small `watercraft` package; move the contract, task, model, logger and CEM implementation. Keep current observations, rewards and checkpoints compatible. Make training and evaluation use independent policy loading.
3. **Move simulator and firmware source.** Relocate the ROS workspace source and PlatformIO projects, fix launch/build paths, and document the working environments. Verify simulator reset/step/close before broader changes.
4. **Extract real control.** Reuse the current vision geometry and motor conversions in `vision.py` and `motors.py`. Implement explicit physical reset/readiness and the real environment. First verify recorded inputs and stopped/bench operation, then a short supervised trial.
5. **Connect the four commands.** Make train/evaluate select sim or real, use the agreed run layout, and record checkpoint-specific evaluations. Demonstrate a short real collection/update cycle with motors stopped during updates.
6. **Add algorithms one at a time.** Add PPO or SAC when needed, with its own checkpoint/resume/export support. Exercise the same environment contract in sim before using it for a physical experiment.

Before declaring this complete: existing seven contract tests pass; exported and training inference agree; simulator episodes have the configured length and fresh reset state; real resets and tracking faults stop correctly; learning pauses do not retain thrust; evaluation never learns; checkpoints resume correctly; and building/running does not dirty generated files in Git.

Current issues to resolve separately from file moves: local `boat.xacro` changes mass to 0.5 kg while leaving larger-boat inertia/geometry; the local 3D vision script lacks its required `math` import; transmitter serial timeout is currently 100 seconds while it continues retransmitting old commands. Preserve the local edits, then review these behavior changes explicitly. Tune a measured boat model and task; do not assume moving files fixes sim-to-real differences.

A modern ROS/Gazebo migration is a later simulator project. It is not required to achieve this simpler organization.
