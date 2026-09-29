# Training instability deep dive (2026-09-27/28)

Scope: the 88-run DR sweep (22 variants x near-goal curriculum ON/OFF x payload compensation off/on),
2 GPU x 2048 envs, PPO (RSL-RL 5.0.1), RNN policy with a state-dependent log-std head
(`state_dependent_std=True`, `noise_std_type="log"`, `clip_actions=1.0`, `entropy_coef=0`, adaptive LR,
`desired_kl=0.008`). Data: monitor-mirrored TensorBoard scalars, checkpoints, osmo logs.

Two separate problems were found. Both are real, and they compound.

---------------------------------------------------------------------------------------------------------
## Problem 1: resumed runs collapse (a resume bug) -- FIXED in 794f5d5909b

### Evidence
- E1 (cross-run, `resume_damage.txt`): 5 near-goal OFF / comp on runs went from 0.97-0.99 terminal success
  to 0.02-0.13 in the first iteration after the monitor resumed them from model_100 (action_noise,
  grp_controller, plug_wrench_force, obs_eef_rot, socket_pos); 3 more dropped 0.17-0.20.
- The checkpoints were fine: the monitor's local eval of the same model_100 scored 0.88-1.00.
- At the resume iteration: surrogate loss 0.19-0.22 (normally ~-0.0005), value loss 5-100x, plug
  orientation-exceeded / dropped terminations 0.01 -> 0.3-0.76.
- Every saved checkpoint stores optimizer lr = 1e-5 (the adaptive schedule's floor).

### Cause
RSL-RL restores the optimizer state (lr = 1e-5) but PPO's `self.learning_rate` restarts at the config
value (5e-4) and the adaptive schedule writes it back into the optimizer after the first KL check. The
first updates after a resume therefore run ~30-50x too fast and wreck a converged policy.

### Experiment (E3, `resume_lr_repro/`): resume the collapsed run's model_100 for 4 short iterations
| iter | unpatched terminal success | patched terminal success | surrogate (unpatched -> patched) |
|---|---|---|---|
| 100 | 0.79 | 0.79 | 0.084 -> -0.002 |
| 101 | 0.74 | 0.96 | 0.0024 -> -0.0014 |
| 102 | 0.23 | 0.95 | 0.0032 -> -0.0020 |
| 103 | 0.01 | 0.94 | 0.0001 -> -0.0028 |

### Fix
`train_rsl_rl.py`: after `runner.load`, set `runner.alg.learning_rate` from the restored optimizer
(commit 794f5d5909b, pushed). All five sweeps' relaunches now use this commit.

---------------------------------------------------------------------------------------------------------
## Problem 2: the near-goal (at-goal) curriculum stalls learning, and the unbounded per-state std blows up

### E1: cross-run metrics (`runs_table.txt`)
| sweep | runs reaching 90% terminal success | median iters to 90% | std blown now / spiked | median terminal success now |
|---|---|---|---|---|
| near-goal ON, comp off | 5 / 22 | 85 | 5 / 3 | 0.36 |
| near-goal ON, comp on | 2 / 22 | 82 | 3 / 3 | 0.53 |
| near-goal OFF, comp off | 20 / 22 | 42 | 0 / 0 | 0.996 |
| near-goal OFF, comp on | 19 / 22 | 42 | 0 / 0 | 0.963 (5 resume victims) |

- Without the curriculum almost every variant solves insertion in ~40 iterations (std shrinks to
  ~0.06). With it, most runs plateau at the success the near-goal spawns give for free (~45%) and then
  slide as those spawns anneal away (0.8 -> 0 by iteration 500).
- In the blown ON runs `Policy/mean_std` reaches 1e4-1e11 from iteration 30-60 on.
- The adaptive LR is pinned at the 1e-5 floor in blown runs (47-94% of iterations): large-std states make
  every update's KL large, so the whole policy effectively stops learning. (The opposite hypothesis --
  LR running away upward -- is rejected: LR sits at its floor during the blow-ups.)

### E2: where is the std large? (`std_probe.jsonl`; stochastic rollouts of checkpoints from near-goal vs
approach spawns, 64 envs x 150 steps, std per state and action dim)
| checkpoint | median std | share of (state, dim) with std > 10 | max std | where |
|---|---|---|---|---|
| ON / comp off / finger_friction (blown) | 0.15-0.30 | 0.06-0.08% | 8.7e8 | both spawn types |
| ON / comp off / socket_pos (blown) | 0.32-0.42 | 2.3-3.5% | 9.5e11 | mostly z action; more when not seated |
| ON / comp on / obs_socket_rot (blown) | 0.44-0.54 | 0.15-0.28% | 2.3e13 | mostly y action |
| ON / comp off / grasp_pos (healthy) | ~0.53 | 0 | 6 | - |
| OFF / comp off / all_off | 0.07 | 0 | ~1 | - |
- "Blown" policies are mostly normal; a heavy tail of rare states gets log-std ~20-30 on one or two
  action dims. The logged mean std is dominated by that tail.
- It is NOT specific to near-goal / seated states (hypothesis rejected); it appears from approach spawns
  too. The curriculum's role is indirect: it produces a long low-signal phase (reward barely depends on
  the action while 80% of episodes start inserted) in which the unbounded log-std head drifts, and
  clipping to +/-1 hides huge std from the returns.

### E4: osmo A/B -- one learned std shared across states (`state_dependent_std=False`)
Sweep `dr_sweep_comp_stdfix` (runs 1306-1310, relaunches 1311/1312 and later), same commit/config as the paired
`dr_sweep_comp` controls, on the variants that blew up there + all_off. PRELIMINARY (2026-09-29 02:50 UTC; the
stdfix runs are at iteration 39-114, slowed by preemptions):

| variant | control (per-state std) max `Policy/mean_std` <= iter 120 | stdfix max | terminal success at matched iter (control -> stdfix) |
|---|---|---|---|
| all_off | 4.6 | 0.99 | it 60: 0.46 -> 0.86 |
| finger_friction | 352 | 0.99 | it 60: 0.59 -> 0.59 |
| obs_eef_rot | 4.9 | 0.99 | it 20: 0.51 -> 0.52 (stdfix only at it 39) |
| obs_socket_rot | 3.4e7 | 0.99 | it 80: 0.66 -> 0.44 |
| osc_stiffness | 58 | 0.99 | it 80: 0.86 -> 0.90, it 100: 0.87 -> 0.92 |

- With one global std, the std decays monotonically (0.9 -> 0.56-0.67 by iteration 60-100) in every run; none
  spikes. Every control spiked above 4 within 120 iterations and three went to 58-3.4e7.
- Success is equal or better at matched iterations in 3 of 4 comparable variants; obs_socket_rot is lower at
  iteration 80 (noisy: the stdfix run dipped 0.63 -> 0.44 at iterations 40-80).
- Verdict so far: the global std removes the blow-up; confirm at iteration >= 200 before switching the main line.

### Separate observation
near-goal OFF / comp on / plug_mass sat at 0% for 160 iterations: its policy parks the plug at the socket
mouth ~9 mm short (local optimum; the comp-off twin and comp-on all_off both solve it). With compensation
the plug's weight no longer helps push it in, so the policy must learn to push; this seed did not.

---------------------------------------------------------------------------------------------------------
## Next steps
1. DONE 2026-09-28: the 28 resume-damaged lineages were relaunched from their pre-resume checkpoints on 794f5d5 (runs 1484-1511).
2. Drop the near-goal curriculum for the main line (or shorten/soften it); OFF learns 2x faster and never
   blew up. If kept, it must be paired with a bounded std.
3. Bound the exploration noise: adopt `state_dependent_std=False` if E4 confirms, or clamp log-std
   (e.g. <= 0) in a custom distribution; consider a small entropy/std penalty.
4. Log std percentiles (median / p99) instead of only the mean, so a heavy tail is visible early.
