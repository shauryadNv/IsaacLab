# DisplayPort task-space DR: handoff (audited 2026-09-30 01:47 UTC)

Domain randomization (DR) with a success-driven curriculum (ADR) for the DisplayPort cable-insertion
task-space env (Flexiv Rizon 4S, operational-space control), aimed at sim-to-real transfer.

- **Code**: branch `shauryad/dp_cable_dr` (fork `github.com/shauryadNv/IsaacLab`), 13 commits on top of
  `3065d9d969c` (`shauryad/dp_cable_ship`). Tooling: this directory. Runbook: [README.md](README.md).
- **Corrected follow-up code**: branch `curiep/dp-cable-adr-followup`; `ddda77447e` fixes ADR, randomization,
  observation-noise, and resume semantics, and `333fc839da` adds the controlled follow-up sweep.
- **Training audit**: the original 109 canonical variants have 74 running, 0 pending and 35 inactive.
  Every one of the 34 previously reported HIGH-priority queued variants had been
  submitted; 33 were later canceled and one remained running. Two canceled lineages
  already had active replacements. Do not bulk-resubmit the inactive set while the
  existing 15-minute monitor owns relaunches.
  A cron job on the original machine monitors them every 15 min, relaunches dead runs, and evaluates
  every new checkpoint. Dashboard at `http://localhost:8765` (via `ssh -L 8765:localhost:8765`).
- **Corrected follow-up runs**: four HIGH-priority seed-42 controls were submitted as
  `isaaclab_train_rsl_rl-{1696..1699}`. They compare DR off, corrected all-on ADR with global versus
  state-dependent policy standard deviation, and corrected dry joint friction. Additional seeds are gated on
  healthy startup and telemetry from these four runs.
- **Main results so far**:
  1. With the near-goal curriculum OFF, 18/22 single-knob variants reach >= 0.9 terminal success with their
     knob at ADR level 50 (full range) by iteration ~300.
  2. `all_on` (every knob together) has not converged in any sweep (0.2-0.35 terminal success).
  3. The near-goal curriculum (80% of episodes spawn partly inserted, annealed to 0 by iteration 500) hurts:
     with it ON only 5/22 runs pass 0.9.
  4. Two blockers were found and fixed: resumed runs collapsed (learning-rate bug), and the plug sagged
     under zero action (payload gravity compensation added).
  5. One blocker is open: the per-state policy std blows up. A global std fixes it in the A/B so far.

The original success counts remain useful for prioritizing experiments, but they are not a corrected reproduction:
several old variants used incorrect grasp sampling, socket-baseline, rotation-noise, or joint-friction semantics.
Use runs 1696-1699 to validate conclusions before expanding the sweep.

---------------------------------------------------------------------------------------------------------

## 1. What the env does now

Task: `IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace` (train) and `...-TaskSpace-ROS-Inference`
(same env, used for eval and deployment). 30 Hz policy, actions are relative pose deltas
(`_ACTION_SCALE` = 0.025 m or rad per unit, clipped to +/-1), OSC without inertial decoupling:

| OSC gains (nominal) | translation | rotation |
|---|---|---|
| stiffness K | 300 N/m | 30 N·m/rad |
| damping D (ratio) | 35 N·s/m (zeta 1.01) | 1.1 N·m·s/rad (zeta 0.10) |

The arm is gravity-free; the plug (30 g) is not. Arm joint PD is zeroed, so the OSC drives pure torque.

### 1.1 How to turn DR on (Hydra overrides)

    env.dr.enabled=true env.dr.adr.enable=true env.dr.<knob>.enable=true [env.dr.<knob>.final=[lo,hi]]
    env.actions.arm_action.payload_gravity_compensation=true        # recommended, see section 3
    env.events.reset_plug_curriculum.params.at_goal_prob=0.0        # near-goal curriculum OFF (recommended)
    agent.policy.state_dependent_std=false                          # global std (recommended, pending A/B)

Every knob is off by default, so the unmodified env is exactly the pre-DR baseline.
Config: `source/isaaclab_tasks/isaaclab_tasks/contrib/deploy/cable_insertion/config/displayport_rizon_4s/domain_rand_cfg.py`;
knob-to-term wiring: `domain_rand.py` in the same folder; terms in `contrib/deploy/mdp/dr_*.py`.

### 1.2 ADR (success-driven curriculum)

- One global level 0..50. Each enabled knob's range is linearly interpolated from `initial` (level 0) to
  `final` (level 50).
- The level goes up one step when the smoothed terminal success rate is > 0.4 **and** at least 5 episodes' worth
  of steps have passed since the last change (`min_episodes_between`). It never goes down (`demote=False`).
- Only episodes that spawned at the approach pose are scored. Near-goal spawns are excluded, so they cannot
  inflate the success rate.
- Each checkpoint has an authoritative `model_N.adr_state.json` sidecar containing its
  level and cumulative step. `adr_state.json` is mutable latest-state telemetry and a
  validated legacy fallback. Resume selects the newest complete pair.
- Knobs: `env.dr.adr.{num_levels,init_level,success_threshold,min_episodes_between,demote}`.
  `init_level=50` pins every knob at its final range.
- `env.dr.at_goal_schedule`: `iteration` (default, pre-DR behavior), `adr` (anneal near-goal spawns on the ADR
  level), or `off`.

### 1.3 Knob reference (current defaults, commit c93192d3619)

Level 0 is used at ADR level 0 or for the whole run when ADR is off; level 50 is the maximum. "per env" means
sampled once per env at every reset unless noted.

| # | Knob | Level 0 | Level 50 | Unit / mode | Applied by | Notes |
|---|---|---|---|---|---|---|
| 1 | `osc_stiffness` | x1.0 | x[0.5, 1.0] | scale on OSC K, per env and axis, **log-uniform** | `randomize_osc_task_gains` (reset) | Softer-only (was x[0.5, 2.0]); stiffer blows up contact, section 5 issue 11 |
| 2 | `osc_damping_ratio` | x1.0 | x[1.0, 1.5] | scale on OSC zeta, per env and axis, log-uniform | same term | More-damped-only (was x[0.5, 1.5]) |
| 3 | `joint_armature` | 0 | [0.1, 0.2] | kg*m^2, absolute, per env and arm joint | `randomize_joint_parameters` | Sim baseline is 0; OSC ignores the mass matrix, so this is true plant mismatch |
| 4 | `joint_friction` | 0 | [0, 0.05] | PhysX dimensionless static = dynamic Coulomb coefficient, per env and arm joint | task-private dry-friction event | Leaves viscous damping unchanged; corrected 32-env scripted insertion reached 78.1% at level 50 with zero drops |
| 5 | `finger_friction` | 0.75 | [0.4, 1.1] | static = dynamic mu, gripper fingers | `AdrRigidBodyMaterial` | Grip is mostly geometric; minor effect |
| 6 | `mating_friction` | 0.001 | 0.001 | plug + socket mu | same | **Identity by default** (near-frictionless on purpose). Sweeps set `final=[0.001,0.5]` in the mating_friction variant only; not in `all_on` |
| 7 | `plug_mass` | x1.0 | x[0.5, 2.0] | scale, per env | `randomize_rigid_body_mass` | 15-60 g. Payload comp follows it |
| 8 | `grasp_pos` | 0 | +/-2 mm per axis | plug offset in the gripper at reset | `set_robot_to_object_grasp_pose` | Grasp is scripted, so grasp error is otherwise unmodeled |
| 9 | `grasp_rot` | 0 | +/-2 deg per axis | roll/pitch/yaw | same (`rot_randomization_range`) | |
| 10 | `socket_pos` | +/-(5, 5, 10) mm | +/-(30, 30, 30) mm | socket pose at reset | `AdrResetRootStateUniform` | **Replaces** the env's default +/-(10, 10, 20) mm when enabled, so level 0 is narrower than the baseline |
| 11 | `socket_rot` | +/-1 deg | +/-5 deg | per axis | same | |
| 12 | `obs_socket_pos` | 0 / 0 | bias 5 mm, noise 0.5 mm | per-episode bias + per-step noise, m | obs noise model (`socket_kp_pos`) | Bias-dominated: one-shot perception error. 5 mm > the 3 mm success radius |
| 13 | `obs_eef_pos` | 0 / 0 | bias 5 mm, noise 0.5 mm | same | `eef_pos` | TCP calibration error |
| 14 | `obs_eef_rot` | 0 / 0 | bias 2 deg, noise 0 | angular perturbation composed on SO(3) | `eef_rot_6d` | |
| 15 | `obs_socket_rot` | 0 / 0 | bias 2 deg, noise 0 | same | `socket_kp_rot_6d` | |
| 16 | `action_noise` | 0 / 0 | bias 0.005, noise 0.01 | action units (x 0.025 m) = 0.125 mm/step bias, +/-0.25 mm noise | `NoisyDelayedOperationalSpaceControllerAction` | |
| 17 | `action_latency` | 0 | [3, 4] steps | whole 33 ms steps, per env | same | 100-133 ms (ROS + RDK path) |
| 18 | `plug_wrench_force` | 0 | [-0.6, 0.6] N per axis | plug frame, resampled every U(0.5, 2.0) s | `apply_external_force_torque` (interval) | Cable tug stand-in; ~0.1 N·m at the wrist. Torque knob dropped |

Removed or never added:
- plug torque knob: realistic torques (~0.01 N·m) were indistinguishable from none.
- arm joint PD gain randomization: a no-op, since joint PD is zeroed; knobs 1-2 are the gains in the loop.
- Sweep groups (manifest): `grp_controller` = knobs 1-4; `grp_sense_act` = knobs 10-17; `all_on` = every knob except 6.

**The 88 original-sweep runs use the OLD gain ranges** (stiffness x[0.5, 2.0], damping x[0.5, 1.5]; their commits
predate c93192d). Only the four `*_osccap` sweeps use the capped ranges above.

---------------------------------------------------------------------------------------------------------

## 2. Commits on `shauryad/dp_cable_dr`

| Commit | What |
|---|---|
| f773711217d | Toggleable DR + ADR (`env.dr`), OSC gain event, ADR material / reset-pose wrappers, noisy/delayed OSC action, `spawned_at_goal` mask |
| c8245cd728f | Persist the ADR level across resume (`adr_state.json` + `load_training_state` hook in `train_rsl_rl.py`) |
| af09997467c | Allocate `episode_succeeded` before the managers are built (ADR crashed at startup) |
| bcb8cac1005 | Symmetric plug wrench range (was always pushing +x+y+z) |
| 7fd0b592bb5 | ADR actually drives socket pose and the at-goal prob; score only approach-pose episodes |
| 19194a8b87b | Re-tuned ranges, log-uniform gains, ADR spacing in episodes (was a fixed 3000 steps = 15 of our episodes) |
| 28950c69823 | Obs-noise bias no longer random-walks across resets (`operation="abs"`) |
| 6d3a6963efd, 3e1fc55f34b | Joint friction 4-5 -> 0.8-1.0 -> 0.7-0.8 N*m; drop the plug torque knob |
| 63cd85f0295 | Restore the env step counter on resume (the at-goal anneal restarted at 0.8 after every relaunch) |
| b221791bbbf | Optional payload gravity compensation in the deploy OSC action |
| 794f5d5909b | Resume RSL-RL runs from the checkpoint's learning rate (resumed runs collapsed) |
| c93192d3619 | OSC gain DR softer-only / more-damped-only |

Changelog fragments: `source/isaaclab_tasks/changelog.d/dp-taskspace-domain-randomization.minor.rst`,
`source/isaaclab_rl/changelog.d/rsl-rl-env-training-state-resume.minor.rst`.

---------------------------------------------------------------------------------------------------------

## 3. Payload gravity compensation (controller fix)

**Symptom.** Under zero action the gripper sank 29 mm in 5 s. The plug's weight is never corrected: the arm
is gravity-free, and the `pose_rel` target is re-based on the measured pose every step, so the OSC never sees
an error to push back on. The real robot is a stiff 1 kHz Cartesian position servo and does not sag
(ROS sets no payload).

**Fix** (`contrib/deploy/mdp/actions.py`, `DeployOperationalSpaceControllerAction._add_payload_gravity_compensation`):
every physics step, add `J^T [m*g_b ; (com_b - ee_b) x m*g_b]` to the OSC joint efforts. Here `m` is the plug's
current mass (follows the plug_mass DR) times `payload_mass_scale`. Config: `payload_gravity_compensation`,
`payload_asset_name` (task-space env sets `"dp_plug"`), `payload_mass_scale`. Off by default. Also applied in the
LEAPP-export proxy path. Raises if `body_offset` is set.

**Verified** (`docs/findings/payload_comp_SUMMARY.md`):
- hold: -28.8 -> +0.05 mm; push, yaw and insertion unchanged (8/8);
- scale 2 rises +32 mm, scale 0 sinks like off (sign and magnitude check);
- per-env masses of 15-60 g all hold within 0.2 mm.

**Alternatives tried, rejected** (`docs/findings/gain_sweep_SUMMARY.md`, `decoupled_search_SUMMARY.md`):

| Alternative | Result |
|---|---|
| Stiffer translation (K 500-2000) | Only reduces the sag (to 11-14 mm) and ejects plugs at contact |
| zeta 0.7 on all axes | Rotation unstable |
| Joint friction 0.2 N*m | Kills the sag but blocks insertion |
| Inertial decoupling (F = Lambda(q) a) | Contact-stable only for translational K <= 200. Best: 100/100/100/3000/3000/3000, zeta 1.0 + comp; removes coasting (8.7 -> 1.5 mm) with the same push and yaw authority. Still needs comp (-7.6 to -11.9 mm without) and blows up at the old DR extremes. Not adopted; would need retraining and real-robot comparison |

Neither OSC mode tracks per-step deltas like the real servo: 10 steps at 0.3 ask for 75 mm; the sim moves ~10 mm.

---------------------------------------------------------------------------------------------------------

## 4. Training runs

osmo pool `isaac-dev-l40-04`, image `nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1`, 2 GPU x 2048 envs (4 x 1024 for
OOM cases), 6000 max iterations, RSL-RL PPO with LSTM actor-critic, adaptive LR (desired KL 0.008).
Swift artifacts: `swift://pdx.s8k.io/AUTH_team-isaac/datasets/shauryad/displayport_insertion_rizon4s/<commit>-dr_<variant><TAG_SUFFIX>[-r<k>]/`.

### 4.1 Sweeps (variants x 4 settings; each sweep = one dir in `sweeps/`)

| Sweep | Variants | Near-goal curriculum | Payload comp | Policy std | Gain DR | Initial runs | Status now (run / queued) |
|---|---|---|---|---|---|---|---|
| `dr_sweep` | 22 | ON | off | per-state | old (x0.5-2 / x0.5-1.5) | 1169-1190 | 20 / 2 |
| `dr_sweep_noatgoal` | 22 | OFF | off | per-state | old | 1199-1220 | 18 / 4 |
| `dr_sweep_comp` | 22 | ON | on | per-state | old | 1233-1253, 1275 | 13 / 9 |
| `dr_sweep_noatgoal_comp` | 22 | OFF | on | per-state | old | 1254-1274, 1276 | 17 / 5 |
| `dr_sweep_comp_stdfix` | 5 | ON | on | **global** | old | 1306-1310 | 2 / 3 |
| `dr_sweep_osccap` | 4 | ON | off | per-state | **capped** | 1320-1323 | 1 / 3 |
| `dr_sweep_noatgoal_osccap` | 4 | OFF | off | per-state | capped | 1324-1327 | 1 / 3 |
| `dr_sweep_comp_osccap` | 4 | ON | on | per-state | capped | 1328-1331 | 1 / 3 |
| `dr_sweep_noatgoal_comp_osccap` | 4 | OFF | on | per-state | capped | 1332-1335 | 2 / 2 |
| `dr_sweep_noatgoal_comp_gstd_osccap` | 7 (4 submitted) | OFF | on | global + A/B | capped | 1696-1699 | 0 / 4 |

- **Variants (22).** `all_off`, `all_on`, 18 single knobs (1-18; `mating_friction` with `final=[0.001,0.5]`),
  `grp_controller`, `grp_sense_act`.
  - stdfix = `all_off`, `finger_friction`, `obs_eef_rot`, `obs_socket_rot`, `osc_stiffness` (the variants that blew
    up in `dr_sweep_comp`).
  - osccap = `osc_stiffness`, `osc_damping_ratio`, `grp_controller`, `all_on`.
- **Priority.** `all_on` / `all_off` are pinned HIGH via `resources/`; everything else is LOW (preemptible).
  - The final counted attempt goes out at HIGH.
  - About 34 runs escalated to HIGH during the 09-28 preemption storm, before preemptions stopped counting.
    They now queue behind the 32-GPU HIGH quota (see pending decision D1).
- **Current commits.** Relaunches of the 5 main sweeps use 794f5d5; the osccap sweeps use c93192d.
- **Launches that no longer matter.** 1121-1142 (1 GPU x 1024, cancelled), 1143-1164 (1 GPU x 4096, CUDA OOM).
- **Resume-damaged runs.** 28 lineages were relaunched 09-28 from their pre-resume checkpoints (runs 1484-1511).
  All attempts from the damaging resume onward are marked `superseded:` in `runs.tsv`.

### 4.2 Where they stand (per sweep; full per-run table in Appendix A)

| Sweep | Median iteration | Runs >= 0.9 terminal success | Runs at ADR 50 |
|---|---|---|---|
| `dr_sweep` (near-goal ON) | 317 | 5/22 | 8 |
| `dr_sweep_noatgoal` | 309 | **18/22** | 20 |
| `dr_sweep_comp` (near-goal ON) | 282 | 4/22 | 7 |
| `dr_sweep_noatgoal_comp` | 262 | 12/22 | 17 |
| `dr_sweep_comp_stdfix` | 61 | 1/5 | 0 |
| four `*_osccap` | 47-51 | 0/4, 3/4, 0/4, 3/4 (near-goal ON / OFF alternating) | 0 |

Reading it:
- Near-goal OFF learns; near-goal ON mostly stalls.
- **Hard variants with near-goal OFF**:
  - `all_on` stays at 0.20-0.24 terminal success (ADR 35-39);
  - `grp_sense_act` 0.47-0.54;
  - comp-on stragglers: `plug_mass` 0.00 (local optimum ~9 mm short of seated), `grp_controller` 0.00 at ADR 45,
    `plug_wrench_force` 0.20, `obs_eef_rot` 0.22.
- **Caveat on the eval column.** Until 2026-09-29 the local rollout eval ignored per-run overrides.
  - Comp-trained policies were scored **without** compensation (biased low).
  - stdfix checkpoints failed to load.
  - Fixed now (section 6); only new checkpoints get correct numbers.

---------------------------------------------------------------------------------------------------------

## 5. Issues found and fixes (symptom -> cause -> fix)

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | `env.dr.*` CLI overrides had no effect | Isaac Lab applies Hydra `env.*` overrides after `__post_init__` | DR expanded in the env constructor (f773711) |
| 2 | Joint-PD gain randomization did nothing | Task-space env zeroes arm PD | Randomize OSC task gains instead (`randomize_osc_task_gains`) |
| 3 | Friction / socket pose never widened with ADR | Stock material and reset-pose terms cache ranges in their constructor | `AdrRigidBodyMaterial`, `AdrResetRootStateUniform` (f773711, 7fd0b59) |
| 4 | at-goal prob stuck at 0.8 under `at_goal_schedule=adr`; ADR climbing on free successes | Event cached its params; near-goal spawns scored as successes | Live params; `spawned_at_goal` mask (7fd0b59) |
| 5 | ADR crashed at startup | `episode_succeeded` allocated after managers were built | af09997 |
| 6 | ADR level reset to 0 on every resume | RSL-RL checkpoints carry no env state | `adr_state.json` sidecar + `load_training_state` hook (c8245cd) |
| 7 | Wrench always pushed the same diagonal | Range (0, 1) sampled per axis | Symmetric (-0.6, 0.6) (bcb8cac) |
| 8 | Obs bias at level 50 measured 6.6 mm vs a 5 mm range | Bias noise `operation="add"` random-walked across resets | `operation="abs"` (28950c6) |
| 9 | **Resumed runs collapsed** (0.97-0.99 -> 0.02-0.13 in one iteration) | `runner.load` restores the optimizer LR (1e-5) but PPO's `learning_rate` restarts at 5e-4 and is written back after the first KL check | Set `alg.learning_rate` from the optimizer (794f5d5). Repro: unpatched 0.79 -> 0.01 in 3 iterations, patched 0.79 -> 0.94 |
| 10 | at-goal anneal restarted after each relaunch | `common_step_counter` starts at 0 in a new process | Restore it from the checkpoint iteration (63cd85f) |
| 11 | **Plugs ejected at osc_stiffness / damping level 50** (4/32) | Nominal translational gains have no stability margin: 1.1x K or 0.9x zeta blows up contact. The requested 1.5x / 0.7x caps did **not** fix it (10-31/32 blow up at the worst corner) | Softer-only / more-damped-only ranges (c93192d); `docs/findings/osc_cap_verify_SUMMARY.md` |
| 12 | Joint friction 4-5 N*m froze the arm | OSC torques for normal pose errors are far below that | 0.7-0.8 N*m (6d3a696, 3e1fc55) |
| 13 | Plug sags 29 mm under zero action | Gravity-free arm + re-based targets; plug weight never corrected | Payload gravity compensation (b221791), section 3 |
| 14 | Policy std explodes (mean std 1e4-1e13) in near-goal-ON runs; LR pinned at 1e-5 | Unbounded state-dependent log-std head develops a heavy tail on rare states. The near-goal phase gives little signal, and +/-1 clipping hides huge std from returns | **Open**: global std A/B (section 7) |
| 15 | Near-goal curriculum ON: 7/44 runs reach 0.9 vs OFF 39/44 (median 42 iterations to 0.9) | Long low-signal phase while 80% of spawns start inserted | Recommend OFF; `docs/findings/std_instability_REPORT.md` |
| 16 | CUDA OOM at 1 GPU x 4096 and for some variants at 2 x 2048 | L40 memory during the PPO update | 2 x 2048 default; monitor falls back to 4 x 1024 once per run |
| 17 | 09-28: mass preemptions (exit 2001) + NCCL watchdog SIGABRT; 32 runs burned their attempt cap | Cluster events | Preemption / NCCL no longer counted (121 + 36 backfilled); 0 runs gave up |
| 18 | Moving runs to HIGH restarted them from scratch | Picked the rank-1 swift folder and missed URL-encoded `model_N%2Ept` | Resubmitted from model_100 (1231/1232); `moved:` reason prefix |
| 19 | Local eval of comp / stdfix runs wrong or failing | `rollout.py` built the nominal task without the run's overrides | Worker now passes `agent.policy.*` and `env.actions.*` overrides (monitor code, 2026-09-29) |
| 20 | Local env: import errors | Isaac Lab needs `moviepy<2`; Newton version mismatch | `docs/isaaclab_ship_newton_rollback.txt` |

---------------------------------------------------------------------------------------------------------

## 6. Infrastructure (all in this directory; runbook in README.md)

- **Sweeps.** `sweeps/<name>/{config.sh, manifest.tsv, submit.sh, resources/, runs.tsv}`.
  - `submit.sh [--dry-run] <variant> [reason]` renders `osmo/dp_cable.yaml`.
  - The yaml clones the fork at `COMMIT`, trains, uploads to swift, and can `resume_from` a swift folder.
- **Monitor** (`monitor/tick.sh`, cron every 15 min), per run:
  - osmo status and log tails;
  - mirrors tfevents, `adr_state.json` and the newest checkpoint;
  - builds `metrics.json`;
  - queues a local rollout eval (nominal task, 16 envs, one video) per new checkpoint;
  - relaunches dead runs from the newest checkpoint.
  - Relaunch rules: max 4 counted attempts; preemptions and NCCL timeouts not counted; OOM -> 4 x 1024 once;
    last attempt at HIGH; human cancels and `HOLD` files respected. `MONITOR_DRY_RUN=1` never submits.
- **Dashboard** (`index.html`, `run.html`).
  - Every run with sweep tags (Curriculum / Payload comp / Policy std / Gain DR), status, priority, iteration,
    train and eval success, ADR level.
  - Overlay up to 3 runs; per-run page with videos, eval-vs-iteration, curves, events, overrides, log tail.
- **Live instance.** `~/workspaces/rl_policy/dr_tools/` on the original machine (cron line
  `*/15 * * * * /home/shauryad/workspaces/rl_policy/run_monitor/tick.sh`). This directory is the same code with
  paths parameterized through `env.sh`, and `runs.tsv` snapshots. Taking over: README section 5.
  **Never run two monitors on the same sweeps**: both would relaunch the same dead run.
- **Local tests** (`dev/`): knob read-back at ADR 0/50, per-knob sweep with videos (optionally with comp and
  scripted insertion), payload-comp checks, gain / decoupling / OSC-cap searches, std probe.
  - Results of each are in `docs/findings/`.
  - Knob sweep with comp + insertion: **54/54 range checks pass**.
  - Scripted-insertion effect at level 50: joint_friction 0%, mating_friction 13-19%, obs position bias 16-19%.
    These mark difficulty for a weak scripted controller, not impossibility.

---------------------------------------------------------------------------------------------------------

## 7. Open questions and next steps

1. **Std A/B** (`dr_sweep_comp_stdfix` vs `dr_sweep_comp`), preliminary at iterations 39-114:
   - Global std never exceeds 0.99 and decays steadily; the per-state controls spiked to 4.6-3.4e7.
   - Success is equal or better at matched iterations in 3 of 4 variants.
   - Confirm at >= 200 iterations, then make `agent.policy.state_dependent_std=false` the default for this task
     (or clamp log-std).
2. **Drop the near-goal curriculum** for the main line (`at_goal_prob=0`), or pair it with a bounded std.
3. **`all_on` does not converge** in any sweep. Options:
   - build up from the groups that do converge (knobs that solve alone at level 50);
   - start ADR from a partial level;
   - lower `success_threshold` spacing, or give `all_on` more iterations;
   - re-run it with the capped gain DR + comp + global std + near-goal OFF (none of the current runs combine all four).
4. **Next main-line sweep:** near-goal OFF, comp ON, global std, capped gain DR, on commit >= c93192d.
   Retire the near-goal-ON sweeps once they have served as controls.
5. `plug_mass` + comp stalls 9 mm short (seed-specific local optimum); rerun with a different seed.
6. Compare decoupled OSC (100/3000, zeta 1.0 + comp) against the real robot's servo response before adopting it;
   it would need retraining.
7. `mating_friction` > 0.001 and obs position bias >= the 3 mm success radius are the hardest single knobs:
   decide whether the real system needs them at full range.
8. Log std percentiles (median / p99), not only the mean, so heavy tails show early.

### Pending decisions (asked, not yet answered)
- **D1**: move the ~34 escalated, pending HIGH runs back to LOW so they start now (preemptible) instead of queueing
  behind the 32-GPU HIGH quota.
- **D2**: relaunch with `agent.save_interval=10` (default 50) so preemptions lose fewer iterations.

---------------------------------------------------------------------------------------------------------

## 8. What must change (summary)

| Item | Where | Status |
|---|---|---|
| Resume keeps the checkpoint learning rate | `isaaclab_rl/.../train_rsl_rl.py` | done (794f5d5) |
| Payload gravity compensation on for training + deployment sim | `env.actions.arm_action.payload_gravity_compensation=true` | done, opt-in; make it the task default once comp sweeps confirm |
| OSC gain DR softer / more-damped only | `domain_rand_cfg.py` | done (c93192d) |
| Near-goal curriculum off | `task_space_env_cfg.py` `at_goal_prob` | recommended, not changed in code |
| Global (state-independent) action std | agent cfg `state_dependent_std` | pending A/B confirmation |
| Eval applies run overrides | `monitor/rollout_worker.py` | done 2026-09-29 |
| `all_on` convergence | curriculum / training recipe | open |

---------------------------------------------------------------------------------------------------------

## Appendix A: per-run snapshot (2026-09-29 02:42 UTC)

"osmo attempt" counts every submission including preemptions (the relaunch cap counts only real failures).
Eval = local rollout success of the newest evaluated checkpoint (16 envs, nominal task). For comp runs, evals
before 2026-09-29 were run without compensation.

| sweep | variant | run id | status | prio | osmo attempt | iter | terminal success | ADR level | local eval success (iter) |
|---|---|---|---|---|---|---|---|---|---|
| dr_sweep | action_latency | 1567 | RUNNING | LOW | 7 | 322 | 0.24 | 0 | 0.00 (300) |
| dr_sweep | action_noise | 1561 | RUNNING | LOW | 7 | 333 | 0.96 | 50 | 1.00 (300) |
| dr_sweep | all_off | 1484 | PENDING | HIGH | 5 | 368 | 0.16 | - | 0.00 (350) |
| dr_sweep | all_on | 1499 | PENDING | HIGH | 5 | 367 | 0.19 | 0 | 0.00 (350) |
| dr_sweep | finger_friction | 1559 | RUNNING | LOW | 6 | 303 | 0.28 | 0 | 0.00 (300) |
| dr_sweep | grasp_pos | 1557 | RUNNING | LOW | 7 | 316 | 0.95 | 50 | 1.00 (300) |
| dr_sweep | grasp_rot | 1490 | RUNNING | LOW | 5 | 290 | 0.30 | 0 | 0.00 (250) |
| dr_sweep | grp_controller | 1565 | RUNNING | LOW | 7 | 309 | 0.87 | 50 | 0.62 (300) |
| dr_sweep | grp_sense_act | 1568 | RUNNING | LOW | 7 | 273 | 0.36 | 25 | 0.75 (250) |
| dr_sweep | joint_armature | 1407 | RUNNING | HIGH | 4 | 299 | 0.97 | 50 | 0.69 (250) |
| dr_sweep | joint_friction | 1558 | RUNNING | LOW | 7 | 280 | 0.31 | 0 | 0.00 (250) |
| dr_sweep | mating_friction | 1389 | RUNNING | HIGH | 4 | 289 | 0.94 | 50 | 0.88 (250) |
| dr_sweep | obs_eef_pos | 1493 | RUNNING | LOW | 5 | 334 | 0.34 | 0 | 0.06 (300) |
| dr_sweep | obs_eef_rot | 1555 | RUNNING | LOW | 6 | 341 | 0.16 | 0 | 0.00 (300) |
| dr_sweep | obs_socket_pos | 1542 | RUNNING | LOW | 6 | 298 | 0.29 | 0 | 0.00 (250) |
| dr_sweep | obs_socket_rot | 1560 | RUNNING | LOW | 7 | 341 | 0.47 | 50 | 0.94 (300) |
| dr_sweep | osc_damping_ratio | 1515 | RUNNING | LOW | 4 | 392 | 0.99 | 50 | 1.00 (350) |
| dr_sweep | osc_stiffness | 1514 | RUNNING | LOW | 6 | 310 | 0.18 | 0 | 0.00 (300) |
| dr_sweep | plug_mass | 1488 | RUNNING | LOW | 5 | 324 | 0.26 | 0 | 0.00 (300) |
| dr_sweep | plug_wrench_force | 1566 | RUNNING | LOW | 7 | 318 | 0.59 | 50 | 0.56 (300) |
| dr_sweep | socket_pos | 1491 | RUNNING | LOW | 5 | 291 | 0.30 | 0 | 0.00 (250) |
| dr_sweep | socket_rot | 1345 | RUNNING | HIGH | 4 | 324 | 0.24 | 0 | 0.00 (300) |
| dr_sweep_noatgoal | action_latency | 1547 | PENDING | HIGH | 4 | 295 | 1.00 | 50 | 1.00 (250) |
| dr_sweep_noatgoal | action_noise | 1444 | RUNNING | LOW | 2 | 321 | 1.00 | 50 | 1.00 (300) |
| dr_sweep_noatgoal | all_off | 1431 | PENDING | HIGH | 3 | 269 | 1.00 | - | 1.00 (250) |
| dr_sweep_noatgoal | all_on | 1287 | RUNNING | HIGH | 3 | 120 | 0.20 | 35 | 0.44 (100) |
| dr_sweep_noatgoal | finger_friction | 1562 | RUNNING | LOW | 3 | 297 | 1.00 | 50 | 1.00 (250) |
| dr_sweep_noatgoal | grasp_pos | 1437 | RUNNING | LOW | 2 | 324 | 0.98 | 50 | 1.00 (300) |
| dr_sweep_noatgoal | grasp_rot | 1452 | RUNNING | LOW | 2 | 309 | 0.99 | 50 | 1.00 (300) |
| dr_sweep_noatgoal | grp_controller | 1454 | RUNNING | LOW | 2 | 314 | 0.97 | 50 | 0.75 (300) |
| dr_sweep_noatgoal | grp_sense_act | 1449 | RUNNING | LOW | 2 | 309 | 0.54 | 50 | 0.56 (300) |
| dr_sweep_noatgoal | joint_armature | 1445 | RUNNING | LOW | 2 | 317 | 1.00 | 50 | 0.81 (300) |
| dr_sweep_noatgoal | joint_friction | 1447 | RUNNING | LOW | 2 | 317 | 1.00 | 50 | 0.56 (300) |
| dr_sweep_noatgoal | mating_friction | 1546 | PENDING | HIGH | 4 | 283 | 1.00 | 50 | 0.94 (250) |
| dr_sweep_noatgoal | obs_eef_pos | 1448 | RUNNING | LOW | 2 | 310 | 0.88 | 50 | 1.00 (300) |
| dr_sweep_noatgoal | obs_eef_rot | 1430 | RUNNING | LOW | 2 | 332 | 0.99 | 50 | 1.00 (300) |
| dr_sweep_noatgoal | obs_socket_pos | 1478 | RUNNING | LOW | 2 | 289 | 0.84 | 50 | 1.00 (250) |
| dr_sweep_noatgoal | obs_socket_rot | 1453 | RUNNING | LOW | 2 | 308 | 1.00 | 50 | 1.00 (300) |
| dr_sweep_noatgoal | osc_damping_ratio | 1443 | RUNNING | LOW | 2 | 393 | 1.00 | 50 | 1.00 (350) |
| dr_sweep_noatgoal | osc_stiffness | 1427 | RUNNING | LOW | 2 | 332 | 1.00 | 50 | 1.00 (300) |
| dr_sweep_noatgoal | plug_mass | 1545 | PENDING | HIGH | 4 | 291 | 1.00 | 50 | 1.00 (250) |
| dr_sweep_noatgoal | plug_wrench_force | 1441 | RUNNING | LOW | 2 | 322 | 1.00 | 50 | 1.00 (300) |
| dr_sweep_noatgoal | socket_pos | 1456 | RUNNING | LOW | 2 | 307 | 1.00 | 50 | 1.00 (300) |
| dr_sweep_noatgoal | socket_rot | 1442 | RUNNING | LOW | 2 | 325 | 1.00 | 50 | 1.00 (300) |
| dr_sweep_comp | action_latency | 1553 | PENDING | HIGH | 4 | 266 | 0.35 | 0 | 0.00 (250) |
| dr_sweep_comp | action_noise | 1552 | PENDING | HIGH | 4 | 266 | 0.94 | 50 | 0.81 (250) |
| dr_sweep_comp | all_off | 1549 | PENDING | HIGH | 2 | 264 | 0.80 | - | 0.81 (250) |
| dr_sweep_comp | all_on | 1554 | PENDING | HIGH | 2 | 264 | 0.35 | 0 | 0.00 (250) |
| dr_sweep_comp | finger_friction | 1479 | RUNNING | LOW | 2 | 290 | 0.70 | 11 | 0.06 (250) |
| dr_sweep_comp | grasp_pos | 1466 | RUNNING | LOW | 2 | 304 | 0.39 | 0 | 0.06 (300) |
| dr_sweep_comp | grasp_rot | 1550 | PENDING | HIGH | 4 | 266 | 0.34 | 0 | 0.00 (250) |
| dr_sweep_comp | grp_controller | 1532 | RUNNING | LOW | 3 | 282 | 0.93 | 50 | 0.50 (250) |
| dr_sweep_comp | grp_sense_act | 1563 | RUNNING | LOW | 4 | 281 | 0.22 | 0 | 0.00 (250) |
| dr_sweep_comp | joint_armature | 1526 | RUNNING | LOW | 3 | 283 | 0.32 | 0 | 0.00 (250) |
| dr_sweep_comp | joint_friction | 1548 | PENDING | HIGH | 4 | 267 | 0.34 | 0 | 0.00 (250) |
| dr_sweep_comp | mating_friction | 1482 | RUNNING | LOW | 2 | 290 | 0.43 | 0 | 0.12 (250) |
| dr_sweep_comp | obs_eef_pos | 1528 | PENDING | HIGH | 6 | 207 | 0.42 | 0 | 0.00 (200) |
| dr_sweep_comp | obs_eef_rot | 1551 | PENDING | HIGH | 4 | 267 | 0.51 | 50 | 0.56 (250) |
| dr_sweep_comp | obs_socket_pos | 1474 | RUNNING | LOW | 2 | 299 | 0.59 | 50 | 0.88 (250) |
| dr_sweep_comp | obs_socket_rot | 1476 | PENDING | HIGH | 4 | 260 | 0.38 | 41 | 0.81 (250) |
| dr_sweep_comp | osc_damping_ratio | 1527 | RUNNING | LOW | 3 | 337 | 0.98 | 50 | 1.00 (300) |
| dr_sweep_comp | osc_stiffness | 1512 | RUNNING | LOW | 2 | 285 | 0.91 | 50 | 0.75 (250) |
| dr_sweep_comp | plug_mass | 1524 | RUNNING | LOW | 3 | 283 | 0.89 | 50 | 0.81 (250) |
| dr_sweep_comp | plug_wrench_force | 1460 | RUNNING | LOW | 2 | 307 | 0.27 | 0 | 0.00 (300) |
| dr_sweep_comp | socket_pos | 1459 | RUNNING | LOW | 2 | 307 | 0.28 | 0 | 0.00 (300) |
| dr_sweep_comp | socket_rot | 1457 | RUNNING | LOW | 2 | 311 | 0.20 | 0 | 0.00 (300) |
| dr_sweep_noatgoal_comp | action_latency | 1556 | PENDING | HIGH | 7 | 189 | 1.00 | 50 | 0.88 (150) |
| dr_sweep_noatgoal_comp | action_noise | 1506 | RUNNING | LOW | 5 | 191 | 0.82 | 50 | 0.75 (150) |
| dr_sweep_noatgoal_comp | all_off | 1480 | PENDING | HIGH | 2 | 268 | 1.00 | - | 1.00 (250) |
| dr_sweep_noatgoal_comp | all_on | 1481 | PENDING | HIGH | 2 | 257 | 0.24 | 39 | 0.38 (250) |
| dr_sweep_noatgoal_comp | finger_friction | 1363 | RUNNING | LOW | 3 | 307 | 1.00 | 50 | 1.00 (300) |
| dr_sweep_noatgoal_comp | grasp_pos | 1469 | RUNNING | LOW | 2 | 300 | 0.98 | 50 | 1.00 (300) |
| dr_sweep_noatgoal_comp | grasp_rot | 1536 | RUNNING | LOW | 3 | 280 | 1.00 | 50 | 1.00 (250) |
| dr_sweep_noatgoal_comp | grp_controller | 1564 | RUNNING | LOW | 6 | 178 | 0.00 | 45 | 0.00 (150) |
| dr_sweep_noatgoal_comp | grp_sense_act | 1510 | RUNNING | LOW | 4 | 210 | 0.47 | 50 | 0.69 (200) |
| dr_sweep_noatgoal_comp | joint_armature | 1538 | RUNNING | LOW | 3 | 281 | 1.00 | 50 | 1.00 (250) |
| dr_sweep_noatgoal_comp | joint_friction | 1362 | RUNNING | LOW | 3 | 310 | 1.00 | 50 | 0.88 (300) |
| dr_sweep_noatgoal_comp | mating_friction | 1477 | RUNNING | LOW | 2 | 300 | 1.00 | 50 | 1.00 (300) |
| dr_sweep_noatgoal_comp | obs_eef_pos | 1503 | RUNNING | LOW | 4 | 206 | 0.86 | 50 | 0.94 (200) |
| dr_sweep_noatgoal_comp | obs_eef_rot | 1504 | RUNNING | LOW | 4 | 210 | 0.22 | 48 | 0.31 (200) |
| dr_sweep_noatgoal_comp | obs_socket_pos | 1423 | PENDING | HIGH | 4 | 202 | 0.88 | 50 | 1.00 (200) |
| dr_sweep_noatgoal_comp | obs_socket_rot | 1539 | RUNNING | LOW | 7 | 136 | 0.99 | 50 | 1.00 (100) |
| dr_sweep_noatgoal_comp | osc_damping_ratio | 1534 | PENDING | HIGH | 4 | 333 | 1.00 | 50 | 1.00 (300) |
| dr_sweep_noatgoal_comp | osc_stiffness | 1470 | RUNNING | LOW | 2 | 301 | 1.00 | 50 | 1.00 (300) |
| dr_sweep_noatgoal_comp | plug_mass | 1429 | RUNNING | LOW | 2 | 345 | 0.00 | 0 | 0.00 (300) |
| dr_sweep_noatgoal_comp | plug_wrench_force | 1508 | RUNNING | LOW | 5 | 183 | 0.20 | 50 | 0.12 (150) |
| dr_sweep_noatgoal_comp | socket_pos | 1535 | RUNNING | LOW | 6 | 196 | 0.85 | 50 | 0.81 (150) |
| dr_sweep_noatgoal_comp | socket_rot | 1537 | RUNNING | LOW | 3 | 280 | 1.00 | 50 | 1.00 (250) |
| dr_sweep_comp_stdfix | all_off | 1411 | PENDING | HIGH | 4 | 60 | 0.86 | - | - |
| dr_sweep_comp_stdfix | finger_friction | 1410 | PENDING | HIGH | 4 | 61 | 0.60 | 0 | - |
| dr_sweep_comp_stdfix | obs_eef_rot | 1397 | RUNNING | HIGH | 4 | 39 | 0.75 | 6 | - |
| dr_sweep_comp_stdfix | obs_socket_rot | 1409 | RUNNING | HIGH | 4 | 83 | 0.45 | 14 | - |
| dr_sweep_comp_stdfix | osc_stiffness | 1541 | PENDING | HIGH | 4 | 114 | 0.91 | 40 | - |
| dr_sweep_osccap | all_on | 1323 | RUNNING | HIGH | 1 | 96 | 0.64 | 0 | 0.00 (50) |
| dr_sweep_osccap | grp_controller | 1413 | PENDING | HIGH | 4 | 47 | 0.79 | 6 | 0.00 (0) |
| dr_sweep_osccap | osc_damping_ratio | 1375 | PENDING | HIGH | 4 | 24 | 0.74 | 12 | 0.00 (0) |
| dr_sweep_osccap | osc_stiffness | 1412 | PENDING | HIGH | 4 | 47 | 0.61 | 0 | 0.00 (0) |
| dr_sweep_noatgoal_osccap | all_on | 1327 | PENDING | HIGH | 1 | - | - | - | - |
| dr_sweep_noatgoal_osccap | grp_controller | 1419 | PENDING | HIGH | 4 | 48 | 0.96 | 10 | 0.00 (0) |
| dr_sweep_noatgoal_osccap | osc_damping_ratio | 1415 | PENDING | HIGH | 4 | 46 | 0.99 | 15 | 0.00 (0) |
| dr_sweep_noatgoal_osccap | osc_stiffness | 1401 | RUNNING | HIGH | 4 | 90 | 1.00 | 32 | 1.00 (50) |
| dr_sweep_comp_osccap | all_on | 1331 | RUNNING | HIGH | 1 | 87 | 0.61 | 0 | 0.00 (50) |
| dr_sweep_comp_osccap | grp_controller | 1421 | PENDING | HIGH | 4 | 47 | 0.56 | 0 | 0.00 (0) |
| dr_sweep_comp_osccap | osc_damping_ratio | 1420 | PENDING | HIGH | 4 | 48 | 0.81 | 21 | 0.00 (0) |
| dr_sweep_comp_osccap | osc_stiffness | 1417 | PENDING | HIGH | 4 | 34 | 0.45 | 0 | 0.00 (0) |
| dr_sweep_noatgoal_comp_osccap | all_on | 1335 | RUNNING | HIGH | 1 | 83 | 0.31 | 29 | 1.00 (50) |
| dr_sweep_noatgoal_comp_osccap | grp_controller | 1403 | RUNNING | HIGH | 4 | 44 | 0.95 | 9 | 0.00 (0) |
| dr_sweep_noatgoal_comp_osccap | osc_damping_ratio | 1422 | PENDING | HIGH | 4 | 55 | 0.98 | 18 | 1.00 (50) |
| dr_sweep_noatgoal_comp_osccap | osc_stiffness | 1418 | PENDING | HIGH | 4 | 48 | 0.95 | 10 | 0.00 (0) |
