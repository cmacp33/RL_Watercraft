# Session notes, 2026-11-16: new boat, resized pool, PPO training, evaluation metrics

Branch `tracy-reorg`. Nothing from this session is committed.

---

## 1. Summary

- **Replaced the simulated boat:** the 227 kg, 5 m WAM-V is now our 0.6 kg, 0.2 m ENPH479 catamaran, built from the SolidWorks export.
- **Resized the pool:** from 30 m × 30 m to **1.30 m × 0.60 m**.
- **Rescaled training:** goals, goal radius, observation scales and reward to match the new boat.
- **Trained a PPO policy** (`ppo_enph479`) on the new boat: 100% success in sim, about 3 s to reach the goal.
- **New evaluation metrics:**
  - per-step distance log and a median distance-vs-step curve;
  - per-step model output and motor command;
  - action smoothness.
- **Started the sim-to-real investigation**, because the policy does not yet work on the real boat. Findings are listed in section 7; none are fixed yet.

---

## 2. New boat model

**New file:** `src/gym_gazebo/envs/ros_ws/src/boat_gazebo/urdf/enph479_boat.xacro`
**Meshes:** `models/enph479_boat/meshes/` (`base_link.STL`, `Left_link.STL`, `Right_link.STL`)

| Property | Value |
|---|---|
| Total mass | 0.6 kg (hull 0.59 kg + 2 × 0.005 kg shafts) |
| Inertia | from the SolidWorks re-export, scaled to 0.59 kg |
| Thrust | **0.03 N** per thruster, forward and reverse |
| Thruster layout | 2 propellers, ±0.048 m from the centreline, shafts 25° down |
| Front of boat | **+X** (export +Y, rotated −90° about Z) |
| Collision | 2 pontoon boxes (0.203 × 0.045 × 0.06 m) plus a deck box |
| ArUco marker | 2 cm ahead of the boat origin, on the deck |

**Physics plugins** (VRX `usv_gazebo_dynamics` and `usv_gazebo_thrust`):
- **Buoyancy:** two cylinders with hull radius 0.036 m, width 0.166 m and length 0.203 m. The boat floats with its keel **2.8 cm** below the water.
- **Drag:** estimated from the hull shape, not measured.
  - surge: xU 0.05, xUU 0.95
  - sway: yV 0.2, yVV 6.6
  - yaw: nR 0.0005, nRR 0.0024
- **Added mass is set to 0.** The plugin's defaults were 26–50× the boat's roll and pitch inertia, which drove the physics to NaN.
- **Propeller spin torque** is capped at 1e-4 N·m by the joint effort limit. The plugin's fixed 2 N·m would flip a 0.6 kg boat.

**Measured in Gazebo:**

| | Value |
|---|---|
| Top speed (forward and reverse) | 0.21 m/s |
| Spin rate at full differential thrust | 0.94 rad/s |
| Coasting | 0.21 → 0.04 m/s in 8 s |
| Tilt while floating | 0.7° |

**Removed:** the old WAM-V (`urdf/boat.xacro` and `models/boat/`), staged as deleted in git.

**Assumptions to check on the real boat:**
- the propellers are at the stern;
- the marker is 2 cm ahead of the boat's centre;
- the real top speed and turn rate (the drag values are estimates).

---

## 3. Environment

**Pool** (`worlds/boat.world`):
- walls at x = ±0.65 m and y = ±0.30 m, 0.405 m tall;
- the boat stops at all four walls (checked in Gazebo).

**Launch** (`launch/boat.launch`): spawn at z = −0.028 m (the floating draft), new boat model, overhead camera at 1.215 m.

**Physics speed:** `real_time_update_rate` 2500, so the sim can run at up to 10× real time.

**Size settings** (`train.json` and the `pool_gym_v0.py` defaults):

| Setting | Old | New |
|---|---|---|
| Goal distance from start | 3–12 m | 0.122–0.486 m |
| Goal bounds (±x, ±y) | 13 m | 0.53 m, 0.18 m (now per axis) |
| Goal radius | 0.3 m | **0.08 m** (matches the real boat's stop distance) |
| Marker offset | 0.5 m | 0.02 m |
| Observation position scale | 20 m | 0.81 m |
| Forward speed scale | 3.0 m/s | 0.3 m/s |
| Yaw rate scale | 2.0 rad/s | 1.0 rad/s |
| Distance reward weight | 2.0 | 50.0 |
| Fixed training goal | (12, 12) | (0.45, 0.15) |
| Spawn reset tolerance | 0.3 m | 0.012 m |

Notes on the goal radius and reward:
- Scaling the goal radius straight to 0.012 m was rejected: it is smaller than the 3 cm simulated sensor noise.
- The distance weight was raised because a full approach earned only about 1 point against about 30 points of step penalty.

---

## 4. PPO structure (`src/watercraft/algorithms/ppo.py`)

### Networks: separate actor and critic, no shared layers
```
Actor (deployed):   obs(6) -> Linear 128 -> ReLU -> Linear 2 -> tanh   = action mean in [-1, 1]
Critic (training):  obs(6) -> Linear 128 -> ReLU -> Linear 1           = state value V(s)
Exploration noise:  learned log-std, one per thruster, not state-dependent
                    initial -0.5 (std 0.61), clamped to [-3, 0.5] (std 0.05-1.65)
```
- **Policy type:** Gaussian, `Normal(tanh(actor(obs)), exp(log_std))`. This is continuous PPO with a diagonal Gaussian, where the squashing is applied to the mean, not to the sample.
- **Acting during training:** the sampled action is clipped to [−1, 1] before it goes to Gazebo. The log-probability stored is that of the unclipped sample.
- **Deployment:** only the actor is exported (`best.npz`, run with NumPy on the boat laptop). It is deterministic: action = tanh(actor(obs)), with no noise.

### Inputs and outputs
- **Observation (6, body frame, each value in [−1, 1]):**
  1. goal ahead ÷ 0.81 m
  2. goal to the left ÷ 0.81 m
  3. sin(heading error)
  4. cos(heading error)
  5. forward speed ÷ 0.3 m/s
  6. yaw rate ÷ 1.0 rad/s
- **Action (2):** left and right thrust in [−1, 1], where ±1 = ±0.03 N.
- **Control period:** 0.1 s, at most 300 steps (30 s) per episode.

### Data collection (on-policy)
- Each iteration collects **whole episodes until at least 4096 steps** are stored.
- **Random goals** each episode, with **domain randomisation** each episode:

| What is randomised | Range |
|---|---|
| Motor gain, each thruster independently | 0.7–1.15 |
| Deadband | 0–0.10 |
| Command delay | 0–1 steps |
| Observation noise | position 3 cm, heading 3°, speed 0.05 m/s, yaw rate 0.1 rad/s |

  Reward and termination always use the true state, not the noisy observation.

### Reward (per step, in `pool_gym_v0.py`)
```
r = 50   × (previous distance − current distance)        [m]
  + 0.5  × (previous heading error − current heading error)  [rad]
  − 0.3  × |yaw-rate observation|                      spin penalty
  − 0.1                                                 step penalty
  + 30   once, when within 0.08 m of the goal           ends the episode
```

### Advantage estimation
- **GAE** with γ = 0.99 and λ = 0.95. Rewards are multiplied by `reward_scale` 0.1 first.
- **Terminal handling:** a reached goal is a true terminal (next value = 0). A timeout is a truncation, so the next value is bootstrapped from the critic.
- Advantages are normalised over each batch (mean 0, std 1).

### Update
| Setting | Value |
|---|---|
| Epochs per iteration | 10 |
| Minibatch size | 64 (about 64 minibatches per epoch) |
| Clipped surrogate | clip range 0.2 |
| Value loss | MSE to the GAE returns, weight 0.5, no value clipping |
| Entropy bonus | 0.0 (off) |
| Optimiser | Adam, learning rate 3e-4, single optimiser for both networks |
| Gradient clipping | 0.5, applied **separately** to actor + log-std and to critic |

### Checkpoints and stopping
- **`best.pth` / `best.npz`:** the actor that collected the best iteration, scored by (success rate, then lowest mean final distance) of the training rollout, which includes exploration noise.
- **`latest.pt`:** the full learner (model, optimiser, RNG states) for resuming.
- **`final.pth` / `final.npz`:** the actor at the end of training.
- **Stopping:** after 360 minutes or `max_iterations`. `stop_success` is 2 (above 100%), so the "solved" early stop never triggers; this looks intended, as a way to run until the time limit.

### Result: run `ppo_enph479`
| | |
|---|---|
| Iterations | 183 |
| Environment steps | 754k |
| Wall time | 3.65 h |
| Training success, iteration 0 | 29% |
| Training success, last iterations | 100% |
| Steps to goal | about 30 (3 s) |
| Exploration std | 0.61 → 0.17 |
| Sim evaluation (20 episodes, fixed goal (0.45, −0.15), randomisation on) | **20/20 reached**, median 38 steps, action smoothness 0.10 |

---

## 5. Evaluation changes (`evaluate.py`, `results.py`)

**New outputs for each evaluation** (in `runs/<run>/evaluations/sim/`):

| File | Contents |
|---|---|
| `<ts>_steps.csv` | one row per step: `episode, step, dist_m, model_left, model_right, motor_left, motor_right, action_change` |
| `<ts>.csv` | per-episode rows; new column `action_smoothness` |
| `<ts>_distance.png` | median distance vs step, 25–75% band, each episode faint, goal radius line |
| `<ts>.json` | new `distance_curve` summary block |
| `<ts>.png` | per-episode panels; now also smoothness and steps per episode |

What the new columns mean:
- **`model_*`** is the policy's raw output.
- **`motor_*`** is the command the thruster actually receives: after evaluation noise and the simulated delay, gain and deadband.

**How the distance curve works:**
- **Hold the last value:** an episode that ends early keeps its final distance up to `max_episode_steps`, so every step's median covers all episodes.
- **Median, not mean:** robust to a few stuck episodes. Mode is unusable for continuous values.
- **No binning needed:** steps are discrete (0.1 s each), so step *k* is the same moment in every episode.
- **Fixed goals only:** if the goals differ between episodes, the curve is skipped, because raw distances from different goals are not comparable.

**Summary numbers** (printed and saved in the `.json`):

| Field | Meaning |
|---|---|
| `success_rate` | fraction of episodes that reached the goal |
| `median_steps_to_goal` | median steps among successful episodes |
| `mean_median_dist_m` | average height of the median curve (area under the curve ÷ steps); lower is better |
| `median_final_dist_m` | median curve's final value |
| `median_action_smoothness` | mean \|step-to-step change in model output\|; 0 = constant, 2 = full flip every step |

**Also added during the session:**
- `evaluate.max_episode_steps` in `evaluate.json` overrides the step limit for an evaluation (`null` keeps the training value).
- It is validated in `config.py`.
- A steps-per-episode panel was added to the plots.

**Metrics discussed for later** (not built):
- path efficiency;
- success rate vs goal radius;
- control effort;
- total rotation;
- wall contacts;
- success by goal location, with random goals.

---

## 6. Files changed

| File | Change |
|---|---|
| `urdf/enph479_boat.xacro` | **new**: boat model and plugins |
| `models/enph479_boat/` | **new**: meshes and `model.config` |
| `urdf/boat.xacro`, `models/boat/` | **deleted**: old WAM-V |
| `launch/boat.launch` | new model, spawn depth, camera height |
| `worlds/boat.world` | 1.30 × 0.60 m pool, faster physics update rate |
| `pool_gym_v0.py` | defaults: marker offset, goal range and bounds, spawn tolerance |
| `configs/train.json` | run name, goal, radius, reward weight, scales, goal ranges |
| `configs/evaluate.json` | run, goal, `max_episode_steps` |
| `src/watercraft/evaluate.py` | per-step log, smoothness, summary, step-limit override |
| `src/watercraft/results.py` | distance curves, summary, plots |
| `src/watercraft/config.py` | validates `evaluate.max_episode_steps` |

**Tests:** 12 pass and 3 fail, all in `tests/test_policy_contract.py`. The failing tests still expect the old boat's observation scales (20 m and 3 m/s) and need updating.

---

## 7. Sim-to-real: findings so far (not yet fixed)

The policy works in sim but not on the real boat. Mismatches found so far:

1. **`deploy.json` loads the wrong policy.** `policy_weights` points to `runs/20261102-181000_300steps/best.npz`, the old 5 m boat policy. It should be `runs/ppo_enph479/checkpoints/best.npz`.
2. **`max_action` 0.4 clips the policy.** In sim the left thruster is above 0.95 in about 65% of steps; on the real boat everything is capped at 0.4, so the boat does something the policy never saw.
3. **Speed and yaw-rate observations are very noisy.**
   - Deployment differentiates raw camera poses over one 0.1 s tick, with no filtering. With about 3 cm of position noise, that gives roughly 0.4 m/s of speed noise.
   - The sim trains with only 0.05 m/s of noise, and the speed scale is only 0.3 m/s.
4. **The real motors don't match the simulated motor model.**
   - The firmware has a dead zone (|command| < 30), a 150 ms start kick at 50, and a 100 ms brake-and-neutral sequence on every direction change.
   - The calibration has a deadband of 40, saturation at 90, a right gain of 1.8, and swapped channels.
   - Sim randomisation covers only gains of 0.7–1.15, deadband ≤ 0.10 and delay ≤ 1 step.
5. **Latency is likely larger than one step.** It adds up camera frame, processing, serial, ESP-NOW and the receiver's 20 ms loop.
6. **The sim dynamics are estimates.** Top speed, turn rate and coasting need measuring on the real boat to fit the drag values.
7. **The policy is lopsided.** The left thruster is almost always at full power, which makes it sensitive to any left/right motor mismatch.

**Suggested next steps:**
- **Measure the real boat:** top speed, spin rate, coasting, latency, and per-motor thrust vs command.
- **Model the real motors in the sim:** firmware dead zone, start kick, reversal delay, and the measured gains.
- **Make deployment observations match the sim:** filter the velocity estimates and set observation noise to measured values.
- **Fix `deploy.json`:** weights, `max_action` and goal.
- **Retrain and re-evaluate** with the new metrics.
