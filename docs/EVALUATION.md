# How evaluation works

There are two separate kinds of evaluation:

1. **Choosing `best` during training.** Each algorithm decides by itself which weights to save as `best.pth` / `best.npz`.
2. **Evaluating after training** (`python -m watercraft.evaluate`). This runs a saved checkpoint in Gazebo and writes the metrics. It works the same way whichever algorithm trained the run.

Use part 2 to compare algorithms or runs. Use part 1 only to understand which checkpoint `best` refers to.

---

## 1. Choosing `best` during training

| | PPO | SAC | Cross-entropy |
|---|---|---|---|
| Scored | after every iteration (≥ 4096 steps) | every `eval_every` iterations | after every batch |
| Episodes scored | that iteration's training episodes | `eval_episodes` separate episodes | that batch's training episodes |
| Exploration noise | **on** (sampled actions) | **off** (the mean action, as on the boat) | on |
| Goals | whatever that iteration drew | the same goals every time (random generator reset to `eval_seed`) | whatever that batch drew |
| Score, compared in order | success rate, then −mean final distance | success rate, then −mean final distance, then −mean steps | success rate, then −mean final distance |
| Early "solved" stop | training success ≥ `stop_success` for `stop_iterations` in a row | evaluation success ≥ `stop_success` for `stop_iterations` evaluations in a row | training success, same rule |

Notes:
- **The score is per iteration, not per step.** Success rate and final distance mean the same thing whatever the goal is, so all three algorithms can choose `best` with random goals.
- **PPO and cross-entropy are scored with noise and on changing goals.** An iteration whose goals happened to be easy can win, so their `best` is somewhat noisy.
- **SAC is scored without noise on fixed goals.** Every checkpoint is compared on identical goals, using exactly the actions the boat would take (`tanh(mean)`).
- **Setting `stop_success` to 2 turns early stopping off,** since a success rate can never reach 2. Training then runs until `minutes` or `max_iterations`.
- **`final` is the weights at the moment training stopped.**

---

## 2. Evaluating after training: `python -m watercraft.evaluate`

### Settings (`configs/evaluate.json` → `evaluate`)

| Key | Meaning |
|---|---|
| `run` | the run folder under `runs/` |
| `checkpoint` | for example `best.npz` or `final.pth`; read from `runs/<run>/checkpoints/` |
| `episodes` | how many episodes to run |
| `seed` | seeds the evaluation, so it can be repeated |
| `noise` | Gaussian noise added to the policy's output (0 means the actions the boat would take) |
| `goal` | a fixed goal `[x, y]` in metres, or `null` |
| `random_goals` | a new random goal every episode (cannot be combined with `goal`) |
| `randomize` | domain randomisation: motor gain, deadband, delay and observation noise |
| `max_episode_steps` | the step limit for each episode; `null` keeps the training value |
| `goal_hold_steps` | steps in a row the boat must stay inside the goal radius to count as reached; `null` keeps the training value (1, first touch, for runs trained before this setting existed) |
| `goal_heading_deg` | goal heading in degrees for runs trained with `policy.heading_mode` `"target"`; `null` uses random headings when `random_goals` is true, otherwise the training value. Ignored for older runs |
| `auto_plot` | write the plots when the evaluation finishes |

The network size, observation scales, task and simulator settings always come from the run's own `config.json`, so they match the weights. Only the options above override them.

### Each episode

1. Reset. Record the starting distance as step 0.
2. Every control step (0.1 s):
   - The policy outputs `model_left` and `model_right` in [−1, 1].
   - `noise` is added if it is set.
   - The environment applies the command. With `randomize` on, it first applies the simulated delay, gain and deadband. The command that reaches the thrusters is logged as `motor_left` and `motor_right`.
   - Log the distance to the goal and `action_change` = mean |model output − previous model output|.
3. The episode ends when the boat has stayed within `goal_radius_m` of the goal for `goal_hold_steps` steps in a row (**reached**) or at `max_episode_steps` (**timeout**). Leaving the radius restarts the count. `steps` therefore includes the hold. For runs trained with `policy.heading_mode` `"target"`, a step only counts if the boat also faces the goal heading within `task.goal_heading_tolerance_deg`; the per-episode CSV then has `goal_heading_deg` and `final_heading_error_deg`.

### Output files (`runs/<run>/evaluations/sim/<timestamp>…`)

| File | Contents |
|---|---|
| `<ts>.csv` | one row per episode: `total_reward`, `steps`, `start_dist_m`, `final_dist_m`, `min_dist_m`, `reached`, `goal_x_m`, `goal_y_m`, `end_reason`, `action_smoothness` |
| `<ts>_steps.csv` | one row per step: `episode`, `step`, `dist_m`, `model_left/right`, `motor_left/right`, `action_change` |
| `<ts>.json` | the full settings used, the checkpoint's SHA-256, `status`, and the `distance_curve` summary below |
| `<ts>.png` | per-episode panels: final distance, reached, steps per episode, smoothness |
| `<ts>_distance.png` | the median distance-to-goal curve (fixed goals only) |

### Metrics

| Metric | How it's computed | Better |
|---|---|---|
| **Success rate** | episodes reached ÷ episodes | higher |
| **Median steps to goal** | median `steps` over the episodes that reached the goal | lower |
| **Distance-vs-step curve** | see below | lower and earlier |
| **Mean of the median curve** (`mean_median_dist_m`) | area under the median curve ÷ steps. One number covering both speed and accuracy | lower |
| **Median final distance** | the median curve at the last step | lower |
| **Action smoothness** | per episode: mean \|Δ model output\| per step, averaged over both thrusters. 0 is constant, 2 is a full flip every step. The summary reports the median over episodes | lower |

### The distance-vs-step curve

1. Build an (episodes × `max_episode_steps` + 1) table of distance to goal.
2. **Hold each episode's last distance until the step limit.** An episode that reaches the goal at step 30 keeps its final distance for steps 31 to `max_episode_steps`. Every step then has a value from every episode, so fast episodes aren't dropped from the later steps.
3. Take the **median across episodes at each step**. No binning is needed, because steps are already discrete. The median is used rather than the mean so that one bad episode doesn't drag the whole curve.
4. The plot shows the median line, the 25–75% band, each episode as a faint line, and the goal radius.

**This needs fixed goals.** With random goals, the start distances differ, so "the distance at step 20" isn't comparable across episodes. The curve and its summary are then skipped. Success rate, steps and smoothness still work with random goals.

---

## Comparing runs or algorithms fairly

- Evaluate every run with the **same** `episodes`, `seed`, `goal`, `randomize`, `noise` and `max_episode_steps`.
- Compare `best` with `best` **and** `final` with `final`. PPO and SAC choose `best` differently (section 1).
- Use a fixed goal for the distance curve. Run a second evaluation with `random_goals: true` to check that the policy works for other goals.
- Turn `randomize` on as well as off: it shows how robust the policy is, which is what matters on the real boat.
- Don't compare training logs (`iterations.csv`) across algorithms. They use means over noisy, changing episodes, and the reward weights may differ between runs.
