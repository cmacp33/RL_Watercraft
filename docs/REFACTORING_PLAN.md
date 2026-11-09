# Capstone refactoring and sim-to-real plan

Date: 2026-11-09. Inspected checkout: `28bfcfea530c604e2d27924707559e61f12bbfb7` plus local changes.

## Recommendation

Use a single monorepo containing an installable Python application, separate ROS packages, ESP32 firmware, versioned experiment configuration, and immutable policy releases. Organize around responsibilities and explicit interfaces: task definition, simulator adapter, policy training, evaluation, perception, and deployment runtime.

The key outcome is that the exact exported policy and preprocessing evaluated in simulation are the ones deployed to the boat. A directory reorganization alone cannot establish that guarantee.

There is no universal industry repository tree or provably perfect plan. This design applies established Python packaging, robotics interface, experiment reproducibility, and release-management practices to this project's size. Keep the current learning algorithm initially; isolate infrastructure changes, physics corrections, and algorithm experiments so their effects can be measured separately.

## Inspection scope and evidence

Inventoried all 2,811 tracked paths. Of these, 819 are under ROS `build`, 32 under ROS `devel`, and 1,653 under the vision `venv`: 2,504 tracked generated/dependency files. The remaining 307 include application code, legacy helpers, simulation plugin sources, assets, experiment results, and package metadata. Ignore rules already cover several of these directories but do not untrack existing files.

Read the active training, evaluation, export, logging, policy contract, vision/control, firmware, simulation environment, launch/world/robot description, and installation paths. Inspected legacy Python modules and plugin integration/source paths. Third-party environments, generated build products, meshes, images, and model binaries were inventoried rather than treated as handwritten source requiring line-by-line review. This is an architecture and workflow review, not a mathematical audit of every vendored hydrodynamics implementation.

Local edits existed in `vision_tracking/src/track_aruco_3d.py`, plus untracked `vision_tracking/src/tempCodeRunnerFile.py`. Preserve these during implementation. No application code was changed for this review.

Verification performed:

- `python3 -m unittest discover -s gym_gazebo/tests -v`: all 7 policy-contract tests passed.
- Compared all 6 existing run-level `.npz` policies to their paired `.pth` tensors on 1,000 seeded random observations per pair. Maximum absolute action difference was approximately `6.035e-7`.
- Parsed project Python modules to inspect imports and definitions. Legacy `utils/ros_utils.py` contains a syntax error; `reraise_impl_py2.py` intentionally uses Python 2 syntax.
- Read all three run configurations and episode CSVs; found one stored evaluation CSV containing one failed episode. No broad, held-out simulation evaluation evidence was present in the inspected run outputs.
- This Mac has Torch, NumPy, OpenCV, and Gymnasium available, but lacks `gym` and `rospy` in the invoked Python environment. Gazebo rollouts, firmware builds, and hardware behavior were not validated.

## Current execution paths

```text
training_scripts/gazebo_pool_v0_xentropy.py
  -> gym.make('GazeboPool-v0')
  -> gym_gazebo/envs/pool_gym_env/pool_gym_v0.py
  -> GazeboEnv launches ROS 1 / Gazebo Classic
  -> boat.xacro + boat.world + USV / wave plugins
  -> best.pth, final.pth, best.npz, final.npz, CSV, TensorBoard

run_model.py
  -> imports network architecture from training script
  -> evaluates PyTorch weights in the same Gym environment

vision_tracking/src/track_aruco_v3.py or track_aruco_3d.py
  -> camera calibration + ArUco pose
  -> finite-difference velocity + policy_contract preprocessing
  -> NumpyPolicy or hand-written controller
  -> action cap + motor calibration
  -> serial ASCII -> transmitter ESP32 -> ESP-NOW -> receiver ESP32 -> ESCs
```

Preserve the useful existing features: shared body-relative observations, physical left/right action convention, marker-offset velocity correction in simulation, finite-value action checks, task-based metrics, per-run outputs, NumPy deployment, marker-loss stopping, firmware watchdogs, and neutral-before-reverse handling.

## Findings that determine the design

| Priority | Evidence | Consequence and required change |
| --- | --- | --- |
| P0 | `boat.xacro` defines base mass 227 kg, meter-scale hull geometry, 200 N forward/reverse thrusters; vision v3 describes a 37 × 27 cm rectangle | The configured simulator and tabletop setup have substantially different scales. Measure the actual deployment boat; create separate legacy and measured robot profiles. Normalizing observations does not make dynamics equivalent. |
| P0 | `GOAL_RADIUS_M=0.5`; hardware scripts stop at 0.08 m; training goals span 3–12 m | Current success does not establish the desired real task. Define reach versus stop-and-hold, speed, tolerance, workspace, and duration per task. Train/evaluate the intended task before deployment. |
| P0 | Training scores rollouts before `optimizer.step()` but saves `best.pth` afterward | The saved model is not the model whose rollout score selected it. Evaluate an immutable post-update snapshot on validation scenarios, then save/promote that exact snapshot. |
| P0 | Training early-stop uses noisy training batch success; `saved_weights/pool_policy.pth` is overwritten with final weights | “Solved” and “latest” do not identify a qualified deployment policy. Separate training checkpoints from validated releases and explicit active deployment selection. |
| P0 | `reset()` does not explicitly publish neutral; batch collection resets before yielding; no guaranteed `env.close()` in train/eval | Old thrust may affect reset and between-batch behavior. Stop before reset/exit, clear actuator queues, synchronize fresh state, and guarantee cleanup. The reset-test comment promises more than reset implements. |
| P0 | `_close()` executes global `killall -9` on ROS/Gazebo processes | Closing one environment can terminate unrelated sessions. Own subprocess groups and shut down only owned processes. |
| P0 | `track_aruco_3d.py` defaults to live serial/policy mode and computes `inside_rectangle` without enforcing it | Put actuation behind explicit arm/disarm state and a workspace supervisor. Displaying validity does not enforce operational limits. |
| P1 | Step unpauses before publishing; uses `rospy.sleep(0.1)`, pause service, then wall sleep 0.003 and cached unstamped `ModelStates` | Step/action/state alignment is approximate, especially at 10× physics rate. Provide acknowledged, fixed-step simulator transitions and freshness guarantees. |
| P1 | `done_check()` runs before incrementing count; CSVs contain 301-step episodes; registration has a separate 3,000-step limit | Use one time-limit owner; exact step-count tests; separate task termination from timeout truncation. |
| P1 | Distance/reward/success are reconstructed from clipped normalized observations | “True distance” can be wrong outside the normalization range. Keep metric raw state independent of policy observations. |
| P1 | Simulation uses ground-truth twist; hardware differentiates noisy camera poses | Add an estimator with timestamps, gap handling, and quality metrics; evaluate estimator-driven simulation as well as ideal state. |
| P1 | Hardware caps actions at 0.4; motor gains/deadband/swap live in Python; receiver adds a 150 ms reverse-neutral window | Evaluate the deployed action processing and measured actuator response, including the reverse state machine. Existing gain/deadband/0–1-step randomization does not reproduce it. |
| P1 | Observation noise is added to normalized goal and heading features independently | This does not model correlated pose errors or noisy velocity differences. Perturb raw measurements and run the actual estimator/preprocessor. |
| P1 | `run_model.py` evaluates Torch weights; hardware runs NumPy export | Add evaluation using the final exported runtime artifact; retain parity tests as a release gate. |
| P1 | No training seed CLI; randomness is split across Torch, NumPy, Gym, simulator/plugin RNGs | Record a root seed and derived component seeds. Store explicit evaluation cases rather than relying on shared RNG consumption. |
| P1 | `NumpyPolicy` checks only two weight dimensions; artifact has no contract/version/robot metadata | Validate all keys, layer dimensions, dtypes, finiteness, and manifest compatibility before activation. |
| P1 | v3 detects on undistorted images but passes original distortion to several pose solves | Enforce an image-geometry contract: raw image + original K/D, or rectified image + matching K/zero D. The newer 3D path already uses zero distortion after rectification. |
| P1 | Camera calibration files contain K/D; calibration scripts use absolute paths | Persist camera identity, image size, board dimensions, quality/error statistics, transforms, and calibration version. Reject mismatched camera resolution/configuration. |
| P1 | ESP-NOW structs duplicated; native `int`, message interpretation by size; no explicit sequence/expiry protocol | Share versioned fixed-width protocol definitions, validate sender/type/length/range, reject stale/replayed command sequences, and log receiver status. Use a synchronized snapshot for the command pair and watchdog metadata. |
| P1 | Firmware already has 300 ms serial and 1,000 ms receiver watchdogs, but no latched arm lifecycle | Preserve watchdogs; measure end-to-end stopping delay, neutral behavior, recovery, and reconnect behavior. Require explicit rearm after a fault. |
| P2 | `setup.py` declares only `gym>=0.2.3`; docs refer to Melodic/Gazebo 9, TensorFlow/Keras, another repository | Create tested dependency profiles and accurate installation instructions. Discover the actual working simulation stack before locking it. |
| P2 | Scripts modify `sys.path`; evaluation imports training implementation | Use an installed package and independent policy definition/runtime modules. |
| P2 | Copied Gym wrappers/spaces/utilities include unused arm code and Python 2 remnants; RealEnv/RealEnvROS2 are stubs | Remove after dependency checks; replace maintained Gym features with Gymnasium. Do not describe the ROS2 stub as functioning deployment support. |
| P2 | Run v2 metadata explicitly says config was overwritten and reconstructed; other runs mark dirty checkout | Preserve provenance uncertainty. New runs need immutable resolved configuration and source snapshots/patches, not only a short Git SHA. |
| P2 | ROS package licenses include TODO/TBD; plugin code has upstream headers | Record upstream revisions, local patches, notices, and clarify project licensing before redistribution. |

P0 means address before claiming the current sim results justify real deployment. P1 is required for a reliable reusable pipeline; P2 supports maintainability and reproducibility.

## Target repository structure

`watercraft` is a proposed package name. Keep one Python distribution initially; optional dependencies and import boundaries avoid installing Torch/ROS on the camera machine.

```text
Capstone/
├── README.md
├── pyproject.toml                 # build metadata, optional deps, CLI, lint/test settings
├── requirements/                 # generated, pinned per supported execution profile
│   ├── training-linux.txt
│   ├── runtime-macos.txt
│   └── development.txt
├── .gitignore
├── .pre-commit-config.yaml
├── .github/workflows/
│   ├── python.yml
│   ├── simulation.yml
│   └── firmware.yml
├── src/watercraft/
│   ├── __init__.py
│   ├── cli/                     # thin argument parsing and dependency assembly
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── export.py
│   │   ├── deploy.py
│   │   ├── calibrate.py
│   │   └── report.py
│   ├── core/                    # no ROS, Torch, serial, or camera imports
│   │   ├── types.py             # stamped state, goal, action, diagnostics
│   │   ├── frames.py            # named frame transforms and angle handling
│   │   ├── observations.py      # versioned pure observation construction
│   │   ├── actions.py           # normalized physical left/right contract
│   │   └── interfaces.py        # small Policy, StateSource, Actuator, SimBackend protocols
│   ├── tasks/
│   │   ├── goal_navigation/
│   │   │   ├── reward.py        # selectable, versioned reward definitions
│   │   │   ├── termination.py   # task success/failure, shared across algorithms
│   │   │   └── metrics.py
│   │   └── sampling.py          # training distribution and valid start/goal generation
│   ├── gym/                     # single home for all Gymnasium integration
│   │   ├── __init__.py
│   │   ├── registration.py      # namespaced environment IDs
│   │   ├── factory.py           # build configured environments for train/eval
│   │   ├── checks.py            # Gymnasium compatibility/environment checks
│   │   ├── envs/
│   │   │   └── pool.py          # Env composing task + simulator backend
│   │   ├── wrappers/            # time limits, observation/action transforms
│   │   │   └── randomization.py # measurement/actuator/physics perturbations
│   │   └── simulation/
│   │       ├── gazebo_classic.py # transitional ROS1 backend
│   │       ├── gazebo_sim.py    # modern backend after qualification
│   │       └── session.py       # lifecycle, readiness, deadlines, owned processes
│   ├── policies/
│   │   ├── architectures/       # network definitions, separate from saved weights
│   │   │   ├── mlp.py
│   │   │   └── actor_critic.py  # introduce when an algorithm needs it
│   │   ├── numpy_policy.py      # lightweight deployed inference
│   │   ├── export.py
│   │   └── artifacts.py         # manifest, compatibility, integrity validation
│   ├── training/
│   │   ├── algorithms/
│   │   │   ├── cem.py           # existing implementation
│   │   │   ├── ppo.py           # future adapter to a maintained implementation
│   │   │   └── sac.py           # future adapter to a maintained implementation
│   │   ├── stopping.py          # training budget / convergence rules
│   │   ├── rollout.py
│   │   ├── checkpoint.py        # resumable optimizer/RNG/elite state
│   │   └── trainer.py
│   ├── evaluation/
│   │   ├── scenarios.py         # immutable scenario suite loading
│   │   ├── runner.py
│   │   ├── metrics.py
│   │   └── reports.py
│   ├── perception/
│   │   ├── camera.py
│   │   ├── aruco.py
│   │   ├── calibration.py
│   │   ├── geometry.py          # extract tested 3D geometry from current local code
│   │   └── estimator.py         # stamped pose -> filtered state/quality
│   ├── control/
│   │   ├── baseline.py          # reusable turn-then-drive baseline
│   │   └── actuator_model.py    # measured nonlinear response and reverse transition
│   ├── hardware/
│   │   ├── serial_transport.py
│   │   ├── protocol.py
│   │   └── motor_calibration.py # explicit device-specific conversion
│   ├── deployment/
│   │   ├── runtime.py           # policy + estimator + supervisor + actuator
│   │   ├── supervisor.py        # arm/fault/stop/goal lifecycle
│   │   └── replay.py            # recorded inputs, shadow mode, no actuation
│   └── experiments/
│       ├── config.py           # typed config, schema validation, resolved snapshot
│       ├── logging.py          # CSV/JSONL + optional TensorBoard
│       └── provenance.py
├── configs/
│   ├── robots/                 # legacy_wamv.yaml, measured_boat.yaml
│   ├── tasks/                  # reach_goal.yaml, stop_at_goal.yaml
│   ├── algorithms/             # cem.yaml, ppo.yaml, sac.yaml
│   ├── policies/               # architecture and policy distribution settings
│   ├── training/               # training budgets, validation cadence
│   ├── experiments/            # compose task + robot + algorithm + policy + seed
│   ├── randomization/          # measured uncertainty ranges
│   ├── evaluation/             # validation, heldout, stress suite definitions
│   └── deployment/             # tank setup, action limits, watchdogs, device references
├── ros/
│   ├── legacy_ws/src/          # transitional Catkin workspace, source only
│   │   └── boat_gazebo/        # existing launch/world/URDF, moved without physics edits
│   └── ros2_ws/src/            # introduced during simulator migration
│       ├── watercraft_description/  # URDF/Xacro, mesh assets, marker transform
│       ├── watercraft_gazebo/       # worlds, Gazebo integration, deterministic step/reset
│       ├── watercraft_bringup/      # ROS2 launch and bridge configuration
│       └── watercraft_interfaces/   # only required custom stamped state/command services
├── firmware/
│   ├── shared/include/watercraft_protocol.h
│   ├── transmitter/            # PlatformIO source/config, board-specific pins
│   ├── receiver/
│   └── tests/                  # parser/state-machine tests and bench procedures
├── third_party/
│   ├── README.md               # upstream URLs, exact revisions, licenses, local patch policy
│   ├── vrx_classic/            # pinned source snapshot, or vcs manifest + reviewed patches
│   └── patches/
├── assets/
│   └── markers/                # print-ready fiducials/board specifications
├── calibrations/
│   └── README.md               # format and manifest references; machine artifacts external
├── tests/
│   ├── unit/
│   ├── contract/               # frames, preprocessing, action mapping, export parity
│   ├── gym/                    # environment API, wrappers, simulator lifecycle/reset/step tests
│   ├── integration/            # cross-component integration tests
│   ├── replay/                 # fixed recorded/synthetic pose-camera fixtures
│   └── hardware/               # explicit hardware test selection; never in default CI
├── tools/                      # dataset capture, migration, developer diagnostics
├── containers/                 # pinned legacy simulator and qualified modern stack
├── docs/
│   ├── architecture.md
│   ├── policy_contract.md
│   ├── training.md
│   ├── evaluation.md
│   ├── deployment.md
│   ├── calibration.md
│   ├── decisions/              # short architecture decision records
│   └── REFACTORING_PLAN.md
├── artifacts/                  # ignored local cache; durable copies in artifact storage
│   ├── runs/<run-id>/          # checkpoints, resolved config, logs, provenance
│   ├── evaluations/<eval-id>/  # policy hash, suite, trajectories, reports
│   ├── policies/<policy-id>/   # immutable exported candidates from any algorithm
│   ├── releases/<release-id>/  # qualified bundles referencing exact policy IDs
│   ├── deployments/<session-id>/ # real-world state, commands, diagnostics and results
│   ├── catalog.json           # generated searchable artifact index
│   └── README.md               # storage conventions only
└── data/                       # ignored captures/logs; small test fixtures live in tests
    └── README.md
```

The tree shows responsibility boundaries; do not create empty abstraction layers or one-line modules solely to match it. Merge small neighboring modules until their responsibilities grow. ROS workspaces are source/build integration boundaries, not nested Python package data. Record external plugin locations in workspace dependency manifests; never maintain duplicate copies of the same plugin in both workspaces.

Use a Python `src` layout and `pyproject.toml` entry points so imports work from an installed wheel and any working directory. The [Python Packaging User Guide](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) explains the isolation benefits of this layout.

## Training, evaluation, and deployment folders

Keep `training/`, `evaluation/`, and `deployment/` as separate sibling packages under `src/watercraft/`. Each owns a distinct execution workflow:

| Folder | Responsibility | Main inputs and outputs |
| --- | --- | --- |
| `training/` | Algorithm-specific data collection, updates, and checkpointing | Experiment configuration and environment -> training checkpoints and candidate policies |
| `evaluation/` | Run frozen policies through simulation scenarios and produce comparable metrics | Exact exported policy, task/robot profile and scenario suite -> trajectories, metrics and reports |
| `deployment/` | Execute a qualified policy on the real boat with supervision | Release, deployment configuration and live stamped state -> gated commands and session logs |

`policies/` owns shared network definitions, policy loading, inference and export. `core/` owns shared observation/action contracts; `tasks/` owns rewards and task termination. Evaluation and deployment must use the same exported policy, preprocessing, action limits and relevant actuator behavior. Neither workflow imports a trainer or duplicates policy inference. Algorithm collection boundaries and training stop rules remain separate from task termination.

Store generated data outside source packages:

```text
artifacts/
├── runs/<run-id>/          # Training checkpoints, configuration and logs
├── policies/<policy-id>/   # Exported trained policies from any algorithm
├── evaluations/<eval-id>/  # Simulation scenarios, results and reports
├── releases/<release-id>/  # Qualified deployment bundles
└── deployments/<session-id>/ # Real-world commands, state, faults and results
```

Each evaluation references the exact policy hash and scenario/configuration versions. Each deployment session references its release/policy hash, robot profile, calibration and firmware identities. This keeps source code, trained policies, simulation results and real-world session records independently traceable. Artifacts are ignored locally and backed by durable storage; source-controlled manifests can reference selected published results.

## One home for Gym integration

All Gymnasium-specific application code lives under `src/watercraft/gym/`: environments, registration, construction, wrappers, randomization integration, environment checks, and the simulator backends/session lifecycle used by those environments. There is no separate top-level `envs/` or `simulation/` Python package. Training and simulation evaluation obtain environments through `watercraft.gym.factory`; algorithms do not implement their own copies of the environment.

Keep reusable task rewards, task termination and metrics in `tasks/`, and policy preprocessing/inference in `core/` and `policies/`. These are shared domain components rather than Gym implementations. Gym wrappers call those shared functions instead of duplicating them. Real-boat `deployment/` uses live perception and hardware directly and does not depend on Gymnasium or launch a simulator.

ROS package source, worlds, meshes and compiled Gazebo plugins remain together under the ROS/third-party build layout because they have their own build system. The Python Gym-to-Gazebo integration is entirely under `gym/simulation/`. Gym-specific automated tests are grouped under `tests/gym/`, following the repository's separate test layout.

## Dependency and runtime boundaries

- `core`, `tasks`, and artifact manifest validation use the standard library/NumPy and validated configuration types. They must import successfully without ROS, Torch, OpenCV, or serial support.
- `policies.architectures` and `training` own Torch; the deployed runtime imports `numpy_policy` without importing the trainer.
- `gym` owns all Gymnasium integration, including environment factories, wrappers, checks and simulator backends. `gym/simulation` owns ROS/Gazebo interactions; shared task/reward and policy code remain independent of Gymnasium.
- `perception` owns OpenCV; `hardware` owns pyserial and device calibration. Hardware conversion runs once after policy action limits.
- CLI commands compose dependencies. No `sys.path` edits, import-time camera windows, serial opening, ROS initialization, or simulator launch.
- Use optional dependency groups such as `train`, `vision`, `hardware`, `dev`; ROS/C++ dependencies remain in ROS manifests and pinned OS/container profiles.
- Keep CSV/JSON and TensorBoard initially. An external experiment server, Kubernetes, separate microservices, and an ONNX migration are unnecessary for the current 6→128→2 network.

## Contracts to stabilize first

### State and coordinate conventions

Define `PlanarState` with metric position, yaw, world-frame velocity, yaw rate, capture/simulation timestamp, frame ID, reference-point ID, validity and quality. Monotonic receive time is separate from capture time. Do not compare timestamps from different clock domains without a mapping.

Use meters, seconds, radians; boat frame +X forward, +Y left, +Z up; positive yaw counterclockwise. Keep optical camera axes explicit and transform once at the perception boundary. These conventions follow [ROS REP-103](https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst).

Decide whether state references `base_link` or the marker origin. Preserve marker-origin behavior in the legacy contract to retain checkpoint compatibility; express `T_base_from_marker` explicitly and use the same reference for goal distance and velocity. A future base-link contract requires a new observation version and qualification.

### Observation and action contract

Preserve existing observation version 1 exactly while moving code:

```text
[goal_forward / 20 m,
 goal_left / 20 m,
 sin(heading_error), cos(heading_error),
 forward_speed / 3 m/s,
 yaw_rate / 2 rad/s]
```

Shape `(6,)`, float32, finite, clipped to `[-1, 1]`; physical left/right action shape `(2,)`, finite, normalized `[-1, 1]`; output activation `tanh`, hidden activation ReLU. Record the marker/body convention, layer widths, control period, normalization values, and action semantics in each artifact manifest.

Changing normalization for the tabletop setup changes what the network sees. Treat that as a new contract and retrain or explicitly requalify; never silently reinterpret an old checkpoint.

Keep raw metric state separately. Reward, workspace checks, success, and reports must not reconstruct truth from clipped features. Report observation saturation frequency.

### Task contract

Version the task independently of observation encoding. Define allowable workspace and hull clearance, start/goal sampling, episode duration, goal tolerance, success speed, required dwell, failure conditions, and termination reason. `reach_goal` can terminate on first entry; `stop_at_goal` requires position/speed/dwell conditions and tests drift after arrival.

Gymnasium returns `(obs, info)` from reset and `(obs, reward, terminated, truncated, info)` from step. Use task termination for success/failure and truncation for the configured horizon. Infrastructure failures should raise a recorded fault rather than masquerade as successful trajectories. Follow the [Gymnasium migration guidance](https://gymnasium.farama.org/introduction/migration_guide/).

### Multiple algorithms, policies, and stopping rules

Task termination belongs to the task, not to CEM, PPO, or SAC. Comparing algorithms on the same task requires the same success/failure definitions and evaluation horizon. A rollout buffer ending after a fixed number of transitions is a data-collection boundary, not an environment terminal event. Algorithm adapters own bootstrapping, recurrent-state resets, replay/rollout buffers, and update schedules; training stopping rules own budgets, validation thresholds, and convergence criteria. Preserve `terminated` and `truncated` separately in collected transitions so each algorithm can handle them correctly.

If an experiment deliberately changes terminal conditions, record a new task variant/configuration. Curriculum stages may select task variants during training, but final comparisons use a fixed evaluation task. Reward variants are also selected/versioned independently of algorithms; algorithm-internal objectives such as entropy bonuses or elite ranking belong in the algorithm implementation.

Keep policy architecture source in `src/watercraft/policies/architectures/`, algorithm implementations/adapters in `training/algorithms/`, and trained weights in artifact storage. A policy architecture is not a trained policy, and a resumable training checkpoint is not necessarily a deployable actor. PPO/SAC training checkpoints may include critics, optimizers, and replay/normalization state; deploy only the required actor and preprocessing. Recurrent or stochastic policies require an explicit state/reset and deterministic-evaluation contract; the current two-layer NumPy exporter must reject unsupported architectures rather than silently accepting them.

Compose experiment configurations from independent task, robot, observation, algorithm, architecture, reward, randomization, and seed settings. Use one validated experiment schema plus algorithm-specific options; do not force unrelated algorithms to share optimizer/buffer details. Add adapters when needed rather than implementing every algorithm in advance.

Use unique run and policy IDs. A generated catalog links each policy to its algorithm, architecture, run/checkpoint, task/reward versions, robot profile, seed, artifact hash, evaluation IDs, and qualification status. Filter the catalog by algorithm/task/seed instead of relying on directory names as metadata. Store all retained candidates under `artifacts/policies/<policy-id>/`; releases identify qualified candidates without overwriting them. Back local artifacts with durable storage and a retention policy; pin published/comparison policies and preserve their manifests, configurations, and reports.

Example comparison: CEM + MLP, PPO + actor-critic, and SAC + actor-critic can train with different collection lengths and update rules, while all three exported actors face the same held-out navigation cases, task termination, action constraints, and deployment preprocessing. Budget comparisons should report environment transitions, simulation time, and compute cost rather than equating algorithm iteration counts.

### Policy and simulator interfaces

Keep a small `Policy.act(observation) -> action` interface for NumPy, Torch validation, and the baseline. Prefer a separate state-controller interface for the baseline if it needs richer metric state; do not reverse-engineer raw distance from clipped observations.

A simulator backend owns `start`, `reset(initial_conditions)`, `step(action, duration)`, and `close`. Return raw truth, sensor measurements, simulation timestamps/step IDs, and diagnostics. Session shutdown is idempotent; every readiness/service/state wait has a wall-clock deadline. For the initial legacy backend, characterize and bound timing; a custom acknowledged stepping service/plugin may be required. In the modern backend, advance an integer number of physics ticks, latch both actuator commands together, and return the state corresponding to the completed tick.

Do not force real hardware into a resettable Gym API. Real deployment is a timed runtime with explicit arm, disarm, fault, goal-complete, and manual recovery states.

## Training, evaluation, and deployment lifecycle

```text
resolved config + source snapshot + seeds
    -> training checkpoints
    -> independent validation of a frozen checkpoint
    -> export exact selected checkpoint
    -> NumPy/Torch parity + exported-runtime sim evaluation
    -> immutable release bundle + qualification report
    -> replay / shadow / bench checks
    -> explicitly armed real deployment
    -> timestamped field logs and rollback to qualified release
```

### Training

Retain the current elite-episode regression implementation under an accurately documented CEM-style label. It ranks trajectories with `episode_score`, not cumulative reward, and regresses actions from retained episodes. Moving reward functions alone will not change selection behavior; document score, elite memory, and objective separately.

Extract rollout collection, elite selection, optimizer update, checkpoint writing, and reporting. Use `torch.no_grad()` for action collection. Support seeded initialization and explicit training limits checked during rollouts, with a maximum episode/watchdog duration. Save resumable checkpoints containing network, optimizer, iteration, exploration schedule, elite memory, RNG states, and environment/task configuration. Resume at episode boundaries unless simulator state can also be restored; document that boundary.

After each chosen validation interval: freeze the post-update model, evaluate without exploration on a validation suite, and select by predefined task metrics. Training reward/success remains diagnostic. A final held-out test is run after selection and is not used repeatedly to tune the checkpoint.

Only after infrastructure and physics baselines stabilize, consider a maintained continuous-control implementation such as PPO or SAC as a separately qualified experiment. Do not combine algorithm replacement with the structural migration.

### Evaluation

Separate three suites: training distribution, validation for selection, and held-out final qualification. Materialize scenarios with explicit start position/yaw/velocity, goal, episode seed, dynamics/actuator parameters, sensor perturbations, and task/robot versions. A shared seed alone is insufficient when randomization draws change the sequence of generated goals.

Include nominal navigation, all goal bearings, near/far goals, random initial headings and velocities, wall-adjacent cases with full hull clearance, battery/motor variations, reverse maneuvers, timestamp jitter, camera delay, dropped/stale frames, pose outliers, marker ambiguity, communication loss, and invalid observations. Quantify dynamics uncertainty from measurements rather than arbitrary broad ranges.

Run distinct modes:

1. Ideal-state simulation for policy/task debugging.
2. Measurement/estimator simulation using the deployed preprocessing, estimator, calibration/model, action cap, and actuator delay/reversal behavior.
3. Vision-in-loop simulation or recorded-camera replay to validate actual ArUco perception. A rendered camera suite is optional for fast training, but ground-truth-only evaluation cannot qualify perception.
4. Hardware-in-loop and supervised real trials for transport/ESC behavior and final transfer.

Use the same `NumpyPolicy` export in release evaluation. Compare against the turn-then-drive baseline on identical cases. Report per-case and aggregate success, termination reasons, final/minimum distance, time to goal, arrival speed, dwell success, overshoot, path length, action saturation/slew, workspace violations, pose dropouts, and step/control timing. Include all failed/aborted cases; do not silently exclude simulator or hardware faults.

Use multiple training seeds (start with 3–5) and a predeclared evaluation sample size; report uncertainty and paired comparisons. Choose release thresholds before observing final results. As an illustrative statistical bound, zero failures in 100 independent trials only puts the one-sided 95% upper failure probability around 3%; it does not establish universal reliability. Scenario dependence and real-world distribution shift limit that inference.

### Release bundle

```text
artifacts/releases/<release-id>/
├── manifest.json               # schema, IDs, hashes, compatibility constraints
├── policy.npz                  # deployed artifact
├── resolved_config.json
├── qualification.json          # suite ID/hash, metrics, faults, runtime versions
├── provenance.json             # full SHA, dirty patch hash, dependency/container versions
└── model_card.md               # task, intended robot, limits, known failures
```

The manifest records observation/action/task schema versions, architecture, normalization, dtype, control interval, robot profile hash, actuator model version, required calibration compatibility, action limits, source/checkpoint identity, and artifact checksums. Put camera/motor calibration and firmware build identities in the deployment session manifest; calibration changes that affect qualified behavior require requalification.

Use atomic writes and unique evaluation IDs; never overwrite a previous report because a filename matches. Separate mutable training status from immutable initial resolved config. Preserve legacy run files and explicitly label reconstructed/missing provenance.

### Real runtime

Use a latest-sample camera pipeline with capture timestamps and bounded buffering. Run control on a monotonic schedule; detect missed deadlines and stale measurements. Timestamp pose, estimated velocity, observation, raw policy action, limited action, calibrated channel commands, transport sequence, receiver acknowledgement/status where available, and fault/stop reason.

An independent supervisor gates every action: valid calibration and frame, finite state, freshness, workspace/hull clearance, policy/robot compatibility, timing, motor bounds, and arm state. Marker loss or invalid geometry stops and faults according to the tested state machine. Do not automatically resume thrust on re-detection or reconnect. Explicit arm remains an operational command, not a recurring human approval prompt during development.

Retain the transmitter/receiver watchdogs and validate their measured response under failure. A neutral command is not an instantaneous physical stop; measure coast distance and select workspace margins and timeouts accordingly. Record manual stop capability and test it independently of the vision/policy loop.

## Simulator platform strategy

Freeze the working ROS1/Gazebo Classic stack first in a reproducible Linux environment. Existing docs are insufficient to identify that working stack. Record ROS distribution, Python, Gazebo, compiler, OS, plugin revisions, dependency resolution, and run commands. Run Linux simulation on a Linux machine or suitable VM/container host; keep native Mac camera/hardware dependencies separate.

Gazebo Classic reached end of life in January 2025 ([official Gazebo notice](https://classic.gazebosim.org/)). Modernization is warranted, but is a separate physics/backend migration after characterization.

Proposed conservative target: Ubuntu 24.04 + ROS2 Jazzy + Gazebo Harmonic, subject to a plugin compatibility spike. Current official guidance also lists the newer Ubuntu 26.04 + ROS2 Lyrical + Gazebo Jetty stack; Jazzy/Harmonic remains an explicitly supported pairing. Choose using confirmed hydrodynamics, bridge, headless rendering, and stepping support, then document the decision. See [official compatibility guidance](https://gazebosim.org/docs/harmonic/ros_installation/).

Do not expect Catkin or Classic `ModelPlugin` sources to run unchanged on Gazebo Sim. Investigate supported buoyancy/hydrodynamics/thruster systems or a pinned maintained marine simulation dependency. Port only required local functionality. Compare acceleration, coast-down, turning, reversal, buoyancy, reset behavior, and reference trajectories on both backends. Establish measured tolerances before acceptance; simulator migration requires model/policy requalification and may require retraining.

## File migration map

| Existing content | Destination / action |
| --- | --- |
| `policy_contract.py` | Split pure observations/actions into `core`; NumPy inference into `policies`; device-specific calibration into `hardware`; preserve v1 behavior and temporary compatibility import. |
| `training_scripts/gazebo_pool_v0_xentropy.py` | `training/algorithms/cem.py`, rollout/trainer/checkpoint modules; network to `policies/architectures/mlp.py`; thin CLI. |
| `training_scripts/export_policy.py` | `policies/export.py` + export CLI; retain legacy weight import tool. |
| `training_scripts/run_logging.py` | `experiments`; keep historical CSV reader compatibility and record schema versions. |
| `run_model.py` | `evaluation/runner.py` + evaluate CLI; support exact exported runtime. |
| `plot_runs.py`, `compare_runs.py` | `evaluation/reports.py` + report CLI; legacy CSV import adapters. |
| `pool_gym_v0.py` | Task/reward to `tasks`, sampling/randomization to dedicated modules, Gym API to `gym/envs/pool.py`, randomization integration to `gym/wrappers/randomization.py`, ROS interactions to `gym/simulation/`. |
| `gazebo_env.py` | `gym/simulation/session.py` and Classic backend; replace process ownership and lifecycle code. |
| `gym_gazebo/__init__.py` | `gym/registration.py`, with explicit namespaced registration, e.g. `Watercraft/Pool-v1`; no simulator side effects. |
| `boat_gazebo` | Source-only `ros/legacy_ws/src/boat_gazebo`, then separate description/simulator/bringup packages on ROS2 migration. Select one canonical URDF/model-generation path; reconcile unused SDF template and duplicate marker assets. |
| USV/wave plugins | Pinned `third_party/vrx_classic` source and patch inventory, integrated into legacy build by dependency configuration. Retain notices and required C++ tests. |
| `nodes/boat_env.py`, `testenv.py` | Archive as legacy experiment after confirming no users; they are incomplete alternative environment implementations. |
| `nodes/controller.py` | Preserve as manual simulator diagnostic or consolidate into baseline/control tooling. |
| `real_env.py`, `real_env_ros2.py` | Remove unused stubs after reference checks; real runtime replaces them. |
| Copied `spaces/`, `wrappers/`, generic `utils/` | Prove no active dependency, then remove; use Gymnasium maintained equivalents. Any required project-specific wrappers belong in `gym/wrappers/`; avoid repairing unused generic arm code. |
| `track_aruco_3d.py` | Preserve local work, extract geometry/perception/estimation/controller/runtime/UI responsibilities with fixtures. |
| `track_aruco_v3.py`, `track_aruco_homo.py` | Retain validated alternative geometry if needed, otherwise archive after replay comparison. Pick a default from measured accuracy and stability, not filename version. |
| Calibration/camera/marker utilities | Import-safe `perception` services + thin calibration/capture CLI/tools; eliminate absolute paths. |
| `camera_calibration.npz`, calibration photos | Calibration artifacts/data manifests; external storage for larger captures; tiny representative fixtures in tests. |
| Marker images/print document | `assets/markers`, with physical print dimensions and dictionary/ID specification. |
| `ESP32_Reciever`, nested transmitter project | `firmware/receiver`, `firmware/transmitter`, shared versioned protocol header; pinned PlatformIO/platform/framework versions. |
| `runs/`, `saved_weights/` | Preserve a hash-indexed legacy archive; new outputs in external/ignored artifacts; explicit qualified release references replace ambiguous latest weights. |
| Tracked venv/build/devel/egg-info | Stop tracking in a dedicated cleanup commit while leaving local files intact. Old Git history remains; history rewriting is unnecessary for this refactor. |
| `.vscode`, temp runner file | Keep minimal portable settings if useful; omit machine-generated include paths and temporary runner files from the target tree. |
| README/INSTALL/setup.py | Accurate quickstarts and role-specific docs; `pyproject.toml` replaces obsolete metadata/dependency instructions. |

## Ordered implementation work packages

Estimates are planning ranges for one engineer familiar with this stack, not commitments. Hardware access, dependency recovery, and hydrodynamics portability can dominate. Each package should produce independently reviewable changes.

| Stage | Work and dependencies | Acceptance / exit criterion | Indicative effort |
| --- | --- | --- | --- |
| 0. Preserve and characterize | Preserve local edits; checksum legacy runs/checkpoints/calibration; capture actual Linux/firmware environments; inventory licenses and measured-vs-assumed robot parameters | Existing quickstarts reproducible; baseline trajectories and tests archived; no loss of experimental evidence | 1–3 days |
| 1. Package and separate | Move Python into `src`, thin CLIs, optional dependencies, legacy shims; move source-only ROS and firmware paths; untrack generated files | Installed wheel works outside repository; no path hacks or import side effects; seven existing tests and all six checkpoint parity comparisons still pass | 2–4 days |
| 2. Correct contracts/lifecycle | Raw metric state, versioned contracts/tasks, exact episode horizon, neutral reset, fresh state, service deadlines, owned process cleanup, Gymnasium transition | Exactly configured step count; repeated reset produces known state with zero thrust; induced startup/service failures exit cleanly; unrelated processes survive close | 3–6 days |
| 3. Reproducible training/evaluation | Seed management, resumable checkpointing, frozen validation snapshots, held-out suites, exported-runtime evaluation, report IDs and provenance | Every score references evaluated weight hash; restart at documented boundary restores training state; repeated explicit scenarios reproduce within declared tolerances | 3–5 days |
| 4. Measured sim-to-real model | Measure hull/marker/thrusters/ESCs/latency/noise; separate robot profiles; actual task/tolerance/action limits; estimator-driven simulation | Open-loop thrust, coast-down and turning/reversal traces fit measured data within predeclared tolerances; uncertainties populate randomization ranges | 3–8 days plus experiment access |
| 5. Deployment runtime/firmware | Extract vision runtime, calibration schema, supervisor, replay/shadow mode, shared protocol, fault logging, watchdog/rearm behavior | Recorded replay is repeatable; injected stale/invalid/lost data leads to neutral within measured bound; bench/HIL tests verify parser, timeout, reverse, disconnect and rearm | 4–7 days plus bench access |
| 6. Modern simulator backend | After baseline contracts and legacy characterization; compatibility spike, ROS2/Gazebo migration, physics regressions | Supported clean build and headless evaluation; measured dynamics and fixed-step tests pass; requalified policies; legacy backend can be retired | Spike 2–3 days; migration 1–3+ weeks |
| 7. Qualify and operate | Retrain for measured robot/task; validation then held-out test; release bundle; supervised real trials and rollback exercise | Exact release hash passes predeclared sim and hardware criteria; field results/limits documented; prior qualified release restorable | 2–5 days plus training/trial time |

Stages 4 and 5 can overlap once interfaces stabilize. Stage 6 can proceed independently after Stage 2/3, but must finish before claiming completion of the modern-platform target. The first reliable release on a characterized legacy backend may precede simulator modernization; label that limitation explicitly.

Separate mechanical file moves from behavior changes. Keep thin compatibility entry points for existing commands until replacement commands are exercised. Version changes that affect observation/task/physics behavior; do not relabel changed results as equivalent to historical runs. Tag reproducible milestones and keep a rollback path for backend and firmware changes.

## Proposed developer commands

These are target interfaces, not currently implemented commands:

```bash
watercraft-train --config configs/training/cem_baseline.yaml --seed 42
watercraft-evaluate --checkpoint <checkpoint> --suite configs/evaluation/validation.yaml
watercraft-export --checkpoint <selected-checkpoint> --output <release-dir>
watercraft-evaluate --release <release-dir> --suite configs/evaluation/heldout.yaml
watercraft-calibrate --camera <camera-id> --output <calibration-dir>
watercraft-deploy --release <release-dir> --config <deployment-config> --dry-run
watercraft-deploy --release <release-dir> --config <deployment-config> --arm
watercraft-report --evaluation <evaluation-id>
```

Deployment defaults to dry-run/disarmed; `--arm` is explicit actuation intent. Keep device paths and peer addresses in deployment/board configuration, with local overrides. Do not hide motor-test commands inside camera-preview utilities.

## Verification and CI

| Layer | Required meaningful checks |
| --- | --- |
| Fast Python CI | Formatting/static checks, unit and contract tests, artifact validation, config validation, install wheel and invoke commands from outside checkout. No ROS import required. |
| Contract fixtures | Rotated/translated frames, marker offset and velocity, yaw wrap, nonfinite values, stale/out-of-order timestamps, clipped observations vs raw metrics, motor swap/gain/deadband/limits, malformed weights. |
| Export parity | Fixed reference observations including boundaries and stored replay observations; Torch vs NumPy; provisional float32 tolerance `1e-6`, reviewed when architecture/backends change. |
| Simulator integration | Readiness/timeouts, exact step/action/state alignment, deterministic seeds within documented tolerance, 300-step timeout, neutral/reset/close, repeated episodes, collision/workspace failures, isolated simultaneous sessions. |
| Training behavior | Elite ranking and empty-elite behavior, resumable state roundtrip, selection score associated with exact immutable checkpoint, separation of validation and held-out cases. |
| Perception replay | Synthetic known geometry plus recorded frames; raw vs rectified camera contract, pose flips/outliers, calibration resolution, marker loss/reacquisition, estimator quality and reset after gaps. |
| Firmware | Both pinned board builds; host-testable parser/reverse/watchdog state machine; invalid/type/length/sequence handling and coherent command snapshots. |
| Bench/HIL | Motor directions and calibration, neutral-before-reverse, watchdog stop/coast delay, process freeze/USB loss/radio loss, partial serial lines, reboot/reconnect and explicit rearm. |
| Release qualification | Exact artifact runtime with realistic estimator/action/actuator paths, fixed held-out suite, uncertainty report, baseline comparison, provenance and compatibility checks. |

Default CI must never open serial devices or arm motors. Run simulator jobs on a suitable Linux runner/container. Keep larger training/qualification sweeps scheduled or explicitly invoked; fast PR tests should detect regressions without training an agent from scratch.

## Decisions needed before implementing behavioral changes

1. Identify the real deployment hull and measure mass, geometry, marker transform, thrust curves, reversal response, and coast distance. The tabletop dimensions in vision configuration do not prove the actual vessel dimensions.
2. Specify the operational task: first arrival, stop at goal, or station keeping; tolerance, speed and dwell; tank boundaries and hull margins. A 0.5 m success radius cannot validate an 0.08 m stop task.
3. Identify the currently working training machine and ROS/Gazebo/Python/plugin versions. Freeze evidence before removing generated workspace artifacts from version control.
4. Determine which camera geometry path is most accurate using replay/ground-truth measurements and preserve current uncommitted 3D work.
5. Predeclare acceptable evaluation metrics and failure bounds for simulation, bench, and supervised water trials. Numerical thresholds require measured capability and task requirements; they cannot be inferred from repository structure.

These decisions do not block packaging, provenance preservation, dependency separation, or the initial contract characterization work.
