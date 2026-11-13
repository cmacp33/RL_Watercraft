# Capstone refactoring plan: simulation, real experiments, and multiple algorithms

> Superseded by [SIMPLE_RESTRUCTURING_PLAN.md](SIMPLE_RESTRUCTURING_PLAN.md). Use that simpler plan for implementation. This document is retained as earlier review notes.

Updated: 2026-11-12. Inspected checkout: `21031ec6ebc4d5332b429347f3fe06e14c59708c` plus local changes. This document proposes future implementation; the commands and modules below do not exist yet.

## Recommendation

Use a single monorepo containing an installable Python application, separate ROS packages, ESP32 firmware, versioned experiment configuration, and immutable policy releases. Organize around responsibilities and explicit interfaces: task definition, simulator adapter, policy training, evaluation, perception, and deployment runtime.

The key outcome is that the exact exported policy and preprocessing evaluated in simulation are the ones deployed to the boat. A directory reorganization alone cannot establish that guarantee.

There is no universal industry repository tree or provably perfect plan. This design applies established Python packaging, robotics interface, experiment reproducibility, and release-management practices to this project's size. Make all four workflows architectural requirements, implementing the real collector before declaring the refactor complete. Keep the current learning algorithm initially; isolate infrastructure changes, physics corrections, and algorithm experiments so their effects can be measured separately.

## Inspection scope and evidence

Re-inventoried the filesystem, excluding `.git`: 3,831 files at the start of this review, including 3,500 generated/dependency files (including egg-info), 59 experiment artifacts, and 272 other paths. These are filesystem counts, including ignored files, rather than the old tracked-path counts. See [REFACTORING_INVENTORY.csv](REFACTORING_INVENTORY.csv) for every inventoried path, classification, byte size, and SHA-256. A checksum records identity, not behavioral validation.

Read the active policy contract, CEM training, export, logging, simulation environment/session, evaluation/reporting, 3D perception/controller, firmware, and reset/contract tests. Structurally inspected the remaining Python modules and ROS/C++ sources, build manifests, launch files, robot/world descriptions, shaders, and editor/configuration files. Images, meshes, checkpoint binaries, installed dependencies, and generated outputs were inventoried as assets/artifacts; they were not each visually or mathematically audited. This is a repository architecture review, not proof that every implementation is correct.

Current local edits in `boat.xacro` and `track_aruco_3d.py` must be preserved. The base mass is locally 0.5 kg rather than the committed 227 kg, with the original inertia and meter-scale geometry still present. The 3D script locally lacks `import math` while calling `math` in its controller. Record these as current issues, not validated physics or a functioning hardware training path.

Verification during this update: `python3 -m unittest discover -s gym_gazebo/tests -v` passed all seven tests. Simulator rollouts, firmware compilation, physical trials, training convergence, and new algorithm support were not validated. Historical export-parity results in the prior plan are not newly verified evidence; rerun parity over all retained policies during migration.

## Required workflow matrix

The two independent choices are **operation** (`train` or `evaluate`) and **backend** (`sim` or `real`). Algorithm selection is a third choice during training; evaluation loads the selected policy artifact through its runtime adapter.

| Workflow | Environment/state | Learning | Reset and scheduling | Required output |
| --- | --- | --- | --- | --- |
| Train in sim | Simulator truth for diagnostics; configured truth or estimator measurements for policy | CEM initially; PPO/SAC adapters when added | Automated reset; controlled simulation ticks; isolated workers only after lifecycle tests | Resumable checkpoint, training data/metrics, frozen validation candidates |
| Evaluate in sim | Same robot/task/action stack as the intended experiment | Disabled; frozen actor and normalization | Explicit held-out scenario suite; reproducible initial conditions | Policy hash, scenario IDs, trajectories, task metrics and failures |
| Train in real | Camera -> geometry -> timestamped estimator -> same observation contract | Physical interaction collects data; updates run on host/learner | Explicit operator reset/readiness; real-time collector; neutral/disarmed between collection and updates | Transitions, interventions, actual initial conditions, policy versions, timing and resumable learner state |
| Evaluate in real | Same collector, estimator, supervisor and actuator path as real training | Disabled; no updates or adaptive normalization | Trial protocol with measured start tolerance, operator readiness, fixed policy | Trial records, calibration/firmware IDs, failures, interventions and uncertainty |

Sim-to-real fine-tuning is a new real-training run with a parent policy/checkpoint reference. Real-to-sim evaluation is allowed when contracts match, but matching tensor shapes alone does not establish robot/task compatibility. Training from scratch on the boat remains an explicit supported experiment, with its own physical-interaction budget and bounded exploration configuration.

Do not create four copies of the project or an algorithm-specific simulator. Use one task implementation, backend adapters, shared collection/runtime services, algorithm-specific learners, and an independent evaluator.

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
| P0 | Committed `boat.xacro` defines base mass 227 kg; local mass is now 0.5 kg but inertia, meter-scale hull geometry and 200 N thruster settings remain; vision v3 describes a 37 × 27 cm rectangle | The configured simulator and tabletop setup have substantially different scales. Measure the actual deployment boat; create separate legacy and measured robot profiles. Normalizing observations does not make dynamics equivalent. |
| P0 | `GOAL_RADIUS_M=0.5`; hardware scripts stop at 0.08 m; training goals span 3–12 m | Current success does not establish the desired real task. Define reach versus stop-and-hold, speed, tolerance, workspace, and duration per task. Train/evaluate the intended task before deployment. |
| P0 | Training scores rollouts before `optimizer.step()` but saves `best.pth` afterward | The saved model is not the model whose rollout score selected it. Evaluate an immutable post-update snapshot on validation scenarios, then save/promote that exact snapshot. |
| P0 | Training early-stop uses noisy training batch success; `saved_weights/pool_policy.pth` is overwritten with final weights | “Solved” and “latest” do not identify a qualified deployment policy. Separate training checkpoints from validated releases and explicit active deployment selection. |
| P0 | `reset()` now pauses, calls reset_world and verifies spawn position via GetModelState, but does not explicitly publish neutral; batch collection resets before yielding; no guaranteed `env.close()` in train/eval | Old thrust may affect reset and between-batch behavior. Stop before reset/exit, clear actuator queues, synchronize fresh state, and guarantee cleanup. The reset-test comment promises more than reset implements. |
| P0 | `_close()` executes global `killall -9` on ROS/Gazebo processes | Closing one environment can terminate unrelated sessions. Own subprocess groups and shut down only owned processes. |
| P0 | `track_aruco_3d.py` defaults to live serial/policy mode and computes `inside_rectangle` without enforcing it | Put actuation behind explicit arm/disarm state and a workspace supervisor. Displaying validity does not enforce operational limits. |
| P1 | Step unpauses before publishing; uses `rospy.sleep(0.1)`, pause service, then wall sleep 0.003 and cached unstamped `ModelStates` | Step/action/state alignment is approximate, especially at 10× physics rate. Provide acknowledged, fixed-step simulator transitions and freshness guarantees. |
| P1 | `done_check()` runs before incrementing count; CSVs contain 301-step episodes; registration has a separate 3,000-step limit | Use one time-limit owner; exact step-count tests; separate task termination from timeout truncation. |
| P1 | Distance/reward/success are reconstructed from clipped normalized observations | “True distance” can be wrong outside the normalization range. Keep metric raw state independent of policy observations. |
| P1 | Simulation uses ground-truth twist; hardware differentiates noisy camera poses | Add an estimator with timestamps, gap handling, and quality metrics; evaluate estimator-driven simulation as well as ideal state. |
| P1 | Hardware caps actions at 0.4; motor gains/deadband/swap live in Python; receiver uses a 50 ms brake + 50 ms neutral reversal sequence, a 150 ms startup kick at command 50, and a minimum running command of 30 | Evaluate the deployed action processing and measured actuator response, including the reverse state machine. Existing gain/deadband/0–1-step randomization does not reproduce it. |
| P1 | Observation noise is added to normalized goal and heading features independently | This does not model correlated pose errors or noisy velocity differences. Perturb raw measurements and run the actual estimator/preprocessor. |
| P1 | `run_model.py` evaluates Torch weights; hardware runs NumPy export | Add evaluation using the final exported runtime artifact; retain parity tests as a release gate. |
| P1 | No training seed CLI; randomness is split across Torch, NumPy, Gym, simulator/plugin RNGs | Record a root seed and derived component seeds. Store explicit evaluation cases rather than relying on shared RNG consumption. |
| P1 | `NumpyPolicy` checks only two weight dimensions; artifact has no contract/version/robot metadata | Validate all keys, layer dimensions, dtypes, finiteness, and manifest compatibility before activation. |
| P1 | v3 detects on undistorted images but passes original distortion to several pose solves | Enforce an image-geometry contract: raw image + original K/D, or rectified image + matching K/zero D. The newer 3D path already uses zero distortion after rectification. |
| P1 | Camera calibration files contain K/D; calibration scripts use absolute paths | Persist camera identity, image size, board dimensions, quality/error statistics, transforms, and calibration version. Reject mismatched camera resolution/configuration. |
| P1 | ESP-NOW structs duplicated; native `int`, message interpretation by size; no explicit sequence/expiry protocol | Share versioned fixed-width protocol definitions, validate sender/type/length/range, reject stale/replayed command sequences, and log receiver status. Use a synchronized snapshot for the command pair and watchdog metadata. |
| P1 | Transmitter currently uses `SERIAL_TIMEOUT_MS=100000` (100 s), resends the last command every 100 ms, and allows its serial failsafe to be toggled; receiver timeout is 1,000 ms; no latched arm lifecycle | Restore a measured host-command freshness bound. Continued retransmission of stale commands prevents the receiver radio timeout from detecting a frozen host. Measure end-to-end stopping delay, neutral behavior, recovery, and reconnect behavior. Require explicit rearm after a fault. |
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
│   ├── envs/
│   │   ├── navigation.py        # shared Gymnasium task/transition assembly
│   │   ├── sim.py               # simulator-specific reset/stepping adapter
│   │   ├── real.py              # operator reset + paced real collector adapter
│   │   ├── factory.py           # validated backend/task/robot composition
│   │   ├── registration.py
│   │   └── randomization.py     # measurement/actuator/physics perturbations
│   ├── simulation/
│   │   ├── gazebo_classic.py    # transitional ROS1 adapter
│   │   ├── gazebo_sim.py        # modern adapter after qualification
│   │   └── session.py           # lifecycle, readiness, deadlines, owned processes
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
│   │   ├── trainer.py           # orchestration; delegate algorithm learning loops
│   │   └── capabilities.py      # validate algorithm/backend/resume/export support
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
│   ├── runtime/                 # shared by real training, evaluation and deployment
│   │   ├── session.py           # policy + estimator + supervisor + actuator
│   │   ├── collector.py         # paced transitions, timestamps, actor version
│   │   ├── reset.py             # operator readiness and reset verification
│   │   ├── actor.py             # immutable inference snapshot handoff
│   │   ├── supervisor.py        # arm/fault/stop/goal lifecycle
│   │   └── replay.py            # recorded inputs, shadow mode, no actuation
│   └── experiments/
│       ├── config.py           # typed config, schema validation, resolved snapshot
│       ├── logging.py          # CSV/JSONL + optional TensorBoard
│       └── provenance.py
├── configs/
│   ├── backends/               # gazebo_classic.yaml, real_camera.yaml
│   ├── robots/                 # legacy_wamv.yaml, measured_boat.yaml
│   ├── tasks/                  # reach_goal.yaml, stop_at_goal.yaml
│   ├── algorithms/             # cem.yaml, ppo.yaml, sac.yaml
│   ├── policies/               # architecture and policy distribution settings
│   ├── training/               # training budgets, validation cadence
│   ├── experiments/            # compose task + robot + algorithm + policy + seed
│   ├── randomization/          # measured uncertainty ranges
│   ├── evaluation/             # validation, heldout, stress suite definitions
│   ├── collection/             # pacing, manual reset, transition validity
│   ├── safety/                 # workspace, exploration envelope, command expiry
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
│   ├── integration/            # real simulator lifecycle/reset/step tests
│   ├── replay/                 # fixed recorded/synthetic pose-camera fixtures
│   └── hardware/               # explicit hardware test selection; never in default CI
├── tools/                      # dataset capture, migration, developer diagnostics
├── containers/                 # pinned legacy simulator and qualified modern stack
├── docs/
│   ├── architecture.md
│   ├── policy_contract.md
│   ├── training.md
│   ├── real_experiments.md     # physical setup/reset, collection, updates, trials
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
│   ├── catalog.json           # generated searchable artifact index
│   └── README.md               # storage conventions only
└── data/                       # ignored captures/logs; small test fixtures live in tests
    └── README.md
```

The tree shows responsibility boundaries; do not create empty abstraction layers or one-line modules solely to match it. Merge small neighboring modules until their responsibilities grow. ROS workspaces are source/build integration boundaries, not nested Python package data. Record external plugin locations in workspace dependency manifests; never maintain duplicate copies of the same plugin in both workspaces.

Use a Python `src` layout and `pyproject.toml` entry points so imports work from an installed wheel and any working directory. The [Python Packaging User Guide](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) explains the isolation benefits of this layout.

## Dependency and runtime boundaries

- `core`, `tasks`, and artifact manifest validation use the standard library/NumPy and validated configuration types. They must import successfully without ROS, Torch, OpenCV, or serial support.
- `policies.architectures` and `training` own Torch; the deployed runtime imports `numpy_policy` without importing the trainer.
- `envs` owns Gymnasium task composition for both sim and real. Simulator and real-runtime adapters own transport, state acquisition, pacing and reset details, not reward or learning logic.
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

Expose a Gymnasium adapter for algorithm compatibility, but do not pretend that hardware supports teleportation or pause. `RealNavigationEnv.reset()` delegates to an explicit operator/reset coordinator with a deadline and verifies fresh pose, acceptable starting motion, workspace and readiness before returning. Evaluation and deployment use the same timed runtime with explicit arm, disarm, fault, goal-complete and manual recovery states. A hardware seed controls experiment sampling, not physical determinism.

## Real training is a first-class workflow

### Environment and runtime responsibilities

`envs/navigation.py` constructs observations, invokes the selected task, and returns the standard transition. `envs/sim.py` advances simulator time; `envs/real.py` waits for the next real control interval through `runtime/collector.py`. Both implement the same algorithm-facing observation/action spaces and termination semantics. Hardware-specific readiness and manual resets remain explicit behind the real adapter.

The shared real path is:

```text
camera capture + calibration -> marker geometry -> stamped state estimator
  -> observation builder -> immutable actor -> requested normalized action
  -> supervisor/action envelope -> motor calibration -> serial -> ESP-NOW -> ESC
  -> next fresh measurement -> task reward/termination -> recorded transition
```

This path serves real training, real evaluation, deployment and shadow replay. Camera preview and overlay rendering subscribe to diagnostics; they must not own policy execution, learner updates or motor transmission. Keep the collector/command freshness mechanism responsive when rendering or optimization is slow. Start with collection followed by learning between episodes/batches; asynchronous actor/learner infrastructure is unnecessary until measured timing requires it.

A minimal backend boundary provides `start`, `prepare_episode`, `advance`, `stop`, and `close`, with explicit capabilities such as `automatic_reset`, `controlled_clock`, `truth_state`, and `parallel_instances`. A task computes reward/success from metric state plus goal and elapsed duration. `PolicyRuntime` supplies `reset_episode()` and `act(observation, deterministic)`; recurrent state is explicit if introduced. Do not make all algorithms implement the CEM elite-batch interface.

### Physical episode lifecycle

1. Load and validate configuration, camera calibration, policy contract, firmware/protocol compatibility and output location while disarmed.
2. Stop commands; request operator repositioning or a separately configured reset controller. Never count reset-controller motion as learned-policy data.
3. Verify a fresh pose, frame/reference point, permitted workspace, start-position tolerance and low motion. Record requested and actual initial state. An operator readiness acknowledgement gates each physical reset; a timeout records an aborted preparation rather than generating a fake transition.
4. Arm for collection. Keep one immutable actor version per episode; PPO also requires the appropriate rollout behavior version. Apply bounded exploration on the collector, before the supervisor and motor calibration. Record all resulting action stages.
5. End on task success/failure, time horizon, operator stop, or infrastructure fault. Immediately request neutral and disarm for reset/learning. Record the end reason independently of the learner's termination mask.
6. Perform learner updates while the boat is neutral/disarmed. CEM may collect several episodes before updating; PPO collects its configured rollout; SAC may update from replay between short physical collections. Record update boundaries and never let a long optimizer call preserve old thrust.
7. Validate the next actor snapshot, save the learner checkpoint atomically, then install the snapshot at an explicit collection boundary. Evaluation always freezes the actor and all normalization statistics.

A fault with missing or stale next state is not an ordinary timeout: close the recording segment with a fault record and omit the incomplete transition from learning. A valid task boundary failure can be terminal. An interruption with a trustworthy next state needs a documented truncation/bootstrap rule; never invent a zero terminal observation or label every intervention as task success. Learner code must preserve the final pre-reset observation rather than an auto-reset wrapper's new initial state.

### Transition and provenance schema

Record `run_id`, `episode_id`, `transition_id`, backend, robot/task/reward/observation versions, policy/behavior hash, goal, raw measured state and quality, optional simulator truth, observation before/after, reward, `terminated`, `truncated`, end reason, elapsed duration, and bootstrap validity. Record capture, receive, inference, command-send and next-measurement times with clock-domain labels.

Keep these action fields distinct: policy request; exploratory/behavior action; supervisor-approved normalized action; calibrated channel command; and receiver-applied command when telemetry actually supplies it. Current firmware does not report receiver application state; mark it unknown rather than treating an ESP-NOW send acknowledgement as actuator feedback. Preserve command sequence/expiry and intervention metadata. Simulator truth is privileged diagnostic data and must not leak into a policy configured for estimator observations.

For standard policy-gradient training, retain the sampled policy action and its behavior log probability; do not replace it with a clipped action and then compute likelihood as though that were the original sample. The action-processing/supervision stack is part of the environment transition. Off-policy adapters must likewise define whether the stored action is the requested environment action or a modeled applied action, and use that definition consistently. Log overrides so heavily supervised data can be analyzed or excluded by a predeclared rule.

Dataset segments include source/config hashes, calibration IDs, firmware build ID, host/runtime versions, operator interventions and parent policy IDs. Log real training reward as measured-state reward; without external truth it is not ground truth. Replay supports perception/controller regression and offline dataset inspection. It does not turn an online CEM/PPO/SAC implementation into an offline RL algorithm.

### Algorithm adapters and capability checks

| Adapter | Collection and learner state | Real-experiment implications | Export |
| --- | --- | --- | --- |
| Existing CEM-style elite imitation | Episode scores, elite memory, actor/optimizer, exploration schedule, RNG | Batch episodes can require many manual resets; preserve the current ranking as algorithm logic, independent of task reward | Current ReLU/tanh NumPy MLP supported after shape/parity checks |
| PPO adapter | On-policy rollouts, behavior log probabilities, values, advantages, actor/critic/optimizer and normalization | Freeze behavior during rollout; support partial final rollout according to library semantics; neutral during updates; do not mix old-policy replay into PPO | Export deterministic actor plus preprocessing/action transforms; reject unsupported runtimes |
| SAC adapter | Off-policy replay, actor, critics/targets, optimizers, entropy state, normalization, RNG | Record physical interaction cost and replay ratio; simulator replay reuse is explicit and labeled by domain/task compatibility | Export deterministic actor with its exact squashing/scaling/preprocessing |
| Turn-then-drive baseline | No learner; metric state controller | Same task, action cap, supervisor, trial suite and logging | Versioned controller configuration |

These are implementation requirements, not claims that PPO/SAC currently exist. Start with a maintained algorithm library adapter rather than reimplementing PPO/SAC. A Gymnasium custom environment is the integration boundary described in the [Stable-Baselines3 custom-environment guide](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html); validate with its environment checker and separate collector timing tests. The [Gymnasium environment API](https://gymnasium.farama.org/api/env/) defines reset and terminated/truncated semantics. Pin the tested dependency version when implementing the adapter.

Use an explicit adapter registry with supported spaces, policy types, checkpoint versions, online/offline support, actor import/export and collection requirements. Reject real vectorization, automatic physical reset, unsupported resume and incompatible exports before collection. Do not advertise that every future algorithm works on the boat merely because it can be selected by name.

## Experiment configuration and fair evaluation

Use ordinary validated YAML/JSON composition initially; a configuration framework is optional. Keep backend, robot dynamics, task, reward, observation, policy architecture, learner, collection and actuation limits independent. A resolved experiment could contain:

```yaml
schema_version: 1
operation: train
backend: real_camera
robot: measured_boat_v1
task: stop_at_goal_v1
reward: navigation_progress_v1
observation: body_relative_v1
algorithm: sac
policy: sac_mlp_v1
seed: 42
collection:
  control_period_s: 0.1
  reset_mode: operator
  actor_swap: episode_boundary
training:
  physical_transition_budget: 3000  # illustrative budget, not a recommendation
  update_between_episodes: true
actuation:
  profile: tank_measured_v1
  start_disarmed: true
initialization:
  parent_policy: P001
```

Store resolved contents and hashes, not just names whose files can change later. Robot profiles include mass, inertia, geometry, drag, thrust and marker transforms together. Environment profiles select sim sensors or real camera/serial without changing task definitions. Separate `legacy_wamv` from a measured small boat; changing only mass is not a validated conversion.

Use distinct train, validation and held-out test suites. Training may select on validation results; do not tune repeatedly against held-out results. Each evaluation record has its own ID and immutable policy/suite hashes, rather than overwriting `eval.csv` in a training directory. Evaluate the exported actor runtime when that is the deployment runtime. Compare a baseline using the same task, measurement path and actuator constraints.

A real suite specifies target/start regions, start tolerances, allowed heading/motion, tank setup, battery/calibration information, repeat count and operator-stop criteria. Record actual start conditions, deviations, aborts and interventions; hardware cannot reproduce a simulator seed exactly. Interleave or randomize policy trial order to reduce battery/session effects. Separate preparation failures from completed trials and publish their counts; do not discard inconvenient failed runs.

Compare algorithms over multiple training seeds, paired simulator scenarios where possible, and repeated physical trials. Report success, time to goal, terminal/closest distance, stopping speed/dwell if relevant, workspace violations, intervention/fault rate, actuator saturation, sample count and uncertainty. Plot training against environment transitions and physical/simulated time plus compute/wall time, not just iteration number. The existing plot's claim that iterations are fair across machines does not extend to different algorithms or batch sizes. Separate training curves from frozen-policy evaluation. These choices follow the separate evaluation and repeated-run guidance in the [Stable-Baselines3 RL guide](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html); the concrete trial protocol is a project design proposal.

## Repository hygiene and artifact ownership

Generated ROS build/devel/install/log output, Python environments/caches/egg-info, PlatformIO builds and generated editor paths are not source. Inventory every tracked ignored path before cleanup, then use targeted `git rm --cached` changes to remove them from the index while preserving local files. The current ignore entries do not remove already tracked content. Add corresponding rules for the new workspace locations when moving them. Verify `git ls-files -ci --exclude-standard` and a build/run/status cycle; do not blanket-untrack genuine calibration fixtures or model assets.

Store large runs, datasets, weights and recordings under ignored local `artifacts/` and `data/`, backed by durable storage with hash manifests. Keep tiny test fixtures, configuration, schemas, source assets and curated artifact references in Git. Preserve legacy run directories and their historical metadata before retiring old paths; newly generated reports must never rewrite the original results. Retention rules must keep data needed to reproduce published comparisons and resume retained runs.


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

Freeze and document the working ROS1/Gazebo Classic Linux environment before restructuring its source workspace. Keep its physics unchanged during mechanical moves. The real camera/serial runtime must work independently on the hardware host without installing ROS or Gazebo.

A modern simulator is a separate backend migration after the task, timing and measurement contracts are characterized. Select the ROS/Gazebo pairing through a plugin compatibility spike against official support documentation at implementation time; this refactor does not require choosing a new distribution now. Catkin packages and Classic ModelPlugin code require deliberate porting/replacement. Compare thrust, acceleration, coast-down, turning, reversal, buoyancy, reset and reference trajectories before accepting a replacement backend, and requalify affected policies.

## File migration map

| Existing content | Destination / action |
| --- | --- |
| `policy_contract.py` | Split pure observations/actions into `core`; NumPy inference into `policies`; device-specific calibration into `hardware`; preserve v1 behavior and temporary compatibility import. |
| `training_scripts/gazebo_pool_v0_xentropy.py` | `training/algorithms/cem.py`, shared collector/trainer/checkpoint modules; network to `policies/architectures/mlp.py`; thin CLI. |
| `training_scripts/export_policy.py` | `policies/export.py` + export CLI; retain legacy weight import tool. |
| `training_scripts/run_logging.py` | `experiments`; keep historical CSV reader compatibility and record schema versions. |
| `run_model.py` | `evaluation/runner.py` + evaluate CLI; support exact exported runtime. |
| `plot_runs.py`, `compare_runs.py` | `evaluation/reports.py` + report CLI; legacy CSV import adapters. |
| `pool_gym_v0.py` | Task/reward to `tasks`, sampling/randomization to dedicated modules, Gym API to `envs/pool`, ROS interactions to simulator adapter. |
| `gazebo_env.py` | `simulation/session.py` and Classic backend; replace process ownership and lifecycle code. |
| `gym_gazebo/__init__.py` | Explicit namespaced registration, e.g. `Watercraft/Pool-v1`; no simulator side effects. |
| `boat_gazebo` | Source-only `ros/legacy_ws/src/boat_gazebo`, then separate description/simulator/bringup packages on ROS2 migration. Select one canonical URDF/model-generation path; reconcile unused SDF template and duplicate marker assets. |
| USV/wave plugins | Pinned `third_party/vrx_classic` source and patch inventory, integrated into legacy build by dependency configuration. Retain notices and required C++ tests. |
| `nodes/boat_env.py`, `testenv.py` | Archive as legacy experiment after confirming no users; they are incomplete alternative environment implementations. |
| `nodes/controller.py` | Preserve as manual simulator diagnostic or consolidate into baseline/control tooling. |
| `real_env.py`, `real_env_ros2.py` | Replace unused stubs with `envs/real.py` over the shared runtime/reset coordinator; neither existing stub implements real training. |
| Copied `spaces/`, `wrappers/`, generic `utils/` | Prove no active dependency, then remove; use Gymnasium maintained equivalents. Avoid repairing unused generic arm code. |
| `track_aruco_3d.py` | Preserve local work, extract geometry/perception/estimation/controller/runtime/UI responsibilities with fixtures. |
| `track_aruco_v3.py`, `track_aruco_homo.py` | Retain validated alternative geometry if needed, otherwise archive after replay comparison. Pick a default from measured accuracy and stability, not filename version. |
| Calibration/camera/marker utilities | Import-safe `perception` services + thin calibration/capture CLI/tools; eliminate absolute paths. |
| `camera_calibration.npz`, calibration photos | Calibration artifacts/data manifests; external storage for larger captures; tiny representative fixtures in tests. |
| Marker images/print document | `assets/markers`, with physical print dimensions and dictionary/ID specification. |
| `ESP32_Reciever`, `ESP32_transmitter` (already flattened; nested editor remnants remain) | `firmware/receiver`, `firmware/transmitter`, shared versioned protocol header; pinned PlatformIO/platform/framework versions. |
| `runs/`, `saved_weights/` | Preserve a hash-indexed legacy archive; new outputs in external/ignored artifacts; explicit qualified release references replace ambiguous latest weights. |
| Tracked venv/build/devel/egg-info | Stop tracking in a dedicated cleanup commit while leaving local files intact. Old Git history remains; history rewriting is unnecessary for this refactor. |
| `.vscode`, temp runner file | Keep minimal portable settings if useful; omit machine-generated include paths and temporary runner files from the target tree. |
| `qq` | Accidental ANSI-colored Git-log dump; remove in cleanup after confirming it has no experimental value. |
| README/INSTALL/setup.py | Accurate quickstarts and role-specific docs; `pyproject.toml` replaces obsolete metadata/dependency instructions. |

## Ordered implementation work packages

Estimates are planning ranges for one engineer familiar with this stack, not commitments. Hardware access, dependency recovery, and hydrodynamics portability can dominate. Each package should produce independently reviewable changes.

| Stage | Work and dependencies | Acceptance / exit criterion | Indicative effort |
| --- | --- | --- | --- |
| 0. Preserve and characterize | Preserve local edits; checksum legacy runs/checkpoints/calibration; capture actual Linux/firmware environments; inventory licenses and measured-vs-assumed robot parameters | Existing quickstarts reproducible; baseline trajectories and tests archived; no loss of experimental evidence | 1–3 days |
| 1. Package and separate | Move Python into `src`, thin CLIs, optional dependencies, legacy shims; move source-only ROS and firmware paths; untrack generated files | Installed wheel works outside repository; no path hacks or import side effects; seven existing tests pass and parity passes for every retained export/checkpoint pair | 2–4 days |
| 2. Correct contracts/lifecycle | Raw metric state, versioned contracts/tasks, exact episode horizon, neutral reset, fresh state, service deadlines, owned process cleanup, Gymnasium transition | Exactly configured step count; repeated reset produces known state with zero thrust; induced startup/service failures exit cleanly; unrelated processes survive close | 3–6 days |
| 3. Reproducible training/evaluation | Seed management, resumable checkpointing, frozen validation snapshots, held-out suites, exported-runtime evaluation, report IDs and provenance | Every score references evaluated weight hash; restart at documented boundary restores training state; repeated explicit scenarios reproduce within declared tolerances | 3–5 days |
| 4. Measured sim-to-real model | Measure hull/marker/thrusters/ESCs/latency/noise; separate robot profiles; actual task/tolerance/action limits; estimator-driven simulation | Open-loop thrust, coast-down and turning/reversal traces fit measured data within predeclared tolerances; uncertainties populate randomization ranges | 3–8 days plus experiment access |
| 5. Deployment runtime/firmware | Extract vision runtime, calibration schema, supervisor, replay/shadow mode, shared protocol, fault logging, watchdog/rearm behavior | Recorded replay is repeatable; injected stale/invalid/lost data leads to neutral within measured bound; bench/HIL tests verify parser, timeout, reverse, disconnect and rearm | 4–7 days plus bench access |
| 5b. Real learning and evaluation | After shared runtime and algorithm contract tests: manual-reset real environment, bounded collection, neutral during updates, learner resume, immutable actor handoff, frozen real evaluator | Replay/fake-device tests pass; supervised short physical collection/update/evaluation cycle records valid transitions, faults and operator interventions; hardware evaluation never updates weights | 3–6 days plus physical access |
| 6. Modern simulator backend | After baseline contracts and legacy characterization; compatibility spike, ROS2/Gazebo migration, physics regressions | Supported clean build and headless evaluation; measured dynamics and fixed-step tests pass; requalified policies; legacy backend can be retired | Spike 2–3 days; migration 1–3+ weeks |
| 7. Qualify and operate | Retrain for measured robot/task; validation then held-out test; release bundle; supervised real trials and rollback exercise | Exact release hash passes predeclared sim and hardware criteria; field results/limits documented; prior qualified release restorable | 2–5 days plus training/trial time |

Stages 4 and 5 can overlap once interfaces stabilize. Stage 6 can proceed independently after Stage 2/3, but must finish before claiming completion of the modern-platform target. The first reliable release on a characterized legacy backend may precede simulator modernization; label that limitation explicitly.

Separate mechanical file moves from behavior changes. Keep thin compatibility entry points for existing commands until replacement commands are exercised. Version changes that affect observation/task/physics behavior; do not relabel changed results as equivalent to historical runs. Tag reproducible milestones and keep a rollback path for backend and firmware changes.

## Proposed developer commands

These target interfaces are not currently implemented. Every experiment file resolves the independent backend, robot, task, observation, algorithm, architecture, collection and safety configuration; CLI overrides are validated and saved.

```bash
watercraft-train --backend sim --algorithm cem --config configs/experiments/sim_cem.yaml --seed 42
watercraft-train --backend sim --algorithm ppo --config configs/experiments/sim_ppo.yaml --seed 42
watercraft-train --backend sim --algorithm sac --config configs/experiments/sim_sac.yaml --seed 42
watercraft-evaluate --backend sim --policy artifacts/policies/P001 --suite configs/evaluation/sim_heldout.yaml
watercraft-train --backend real --algorithm sac --config configs/experiments/real_sac.yaml --initialize-from artifacts/policies/P001 --arm
watercraft-train --backend real --algorithm cem --config configs/experiments/real_cem.yaml --arm
watercraft-evaluate --backend real --policy artifacts/policies/P002 --suite configs/evaluation/real_trials.yaml --arm
watercraft-train --resume artifacts/runs/R002/checkpoints/latest
watercraft-export --checkpoint artifacts/runs/R002/checkpoints/selected --output artifacts/policies/P002
watercraft-calibrate --camera tank_camera --output calibrations/local/tank_camera
watercraft-deploy --policy artifacts/policies/P002 --config configs/deployment/tank.yaml --dry-run
watercraft-report --evaluation artifacts/evaluations/E002
```

Real commands default to disarmed preview; `--arm` expresses physical-actuation intent and still requires runtime readiness. Resuming never silently rearms. `--initialize-from` imports a compatible actor into a new run; `--resume` restores the same algorithm's full learner state. Reject incompatible architecture, preprocessing, action semantics or backend capabilities before opening devices. A NumPy actor export is generally insufficient to resume PPO/SAC training; algorithm-specific actor import is a separate validated capability.

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
| Real collection/learning | Fake-clock/recorded-state tests verify action-to-next-state alignment, no stale transitions, operator reset timeout, neutral during optimizer stalls, immutable actor swap and behavior log probabilities; verify checkpoint/resume without auto-rearm. |
| Frozen real evaluation | Same runtime as real training, no gradient/normalization updates, consistent trial schema, preserved failures/interventions, policy/calibration/firmware hashes. |
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
