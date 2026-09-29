# Knob sweep with payload compensation + insertion excitation (2026-09-27)

Setup: `dev/dr_knob_sweep.sh` with `SWEEP_COMP=1 SWEEP_EXCITE=insert`, commit b221791 (payload gravity
compensation ON). One knob enabled at a time, ADR level 0 / 25 / 50, 32 envs, 150 steps (5 s) per level.
Motion: scripted insertion driving the plug mate point onto the socket mate point (|action| <= 0.3); for
obs_socket_pos / obs_eef_pos the controller steers on the NOISY observed position, like a policy would. The
controller is position-only, so it does not correct rotation (socket_rot, grasp_rot) and ignores the
rotation observations (obs_*_rot).

Videos: `<knob>_levels.mp4` (level 0 | 25 | 50) and `all_knobs_levels.mp4` (90 s). Raw: `results.jsonl`.

## Applied correctly: 54/54 PASS
Every knob's sampled values match the expected ADR range at every level (same checks as the original sweep).

## Effect on insertion (success fraction / mean steps seated of 150 / plugs ejected)
| knob | level 0 | level 25 | level 50 | note |
|---|---|---|---|---|
| osc_stiffness | 1.00 / 97 / 0 | 1.00 / 94 / 0 | **0.72 / 58 / 4** | stiffness up to 2x makes contact violent (plugs ejected) |
| osc_damping_ratio | 1.00 / 97 / 0 | 1.00 / 88 / 0 | 0.97 / 86 / **4** | damping down to 0.5x: same contact instability |
| joint_armature | 1.00 / 97 / 0 | 1.00 / 82 / 0 | 1.00 / 82 / 0 | slower, still inserts |
| joint_friction | 1.00 / 97 / 0 | **0.00 / 0 / 0** | **0.00 / 0 / 0** | 0.35-0.8 N*m stops a 0.3-scale controller short of the socket |
| finger_friction | 1.00 / 97 / 0 | 1.00 / 95 / 0 | 0.94 / 80 / 0 | minor |
| mating_friction | 1.00 / 97 / 0 | **0.13 / 9 / 0** | **0.19 / 15 / 0** | plug jams at the socket mouth |
| plug_mass | 0.97 / 93 / 1 | 1.00 / 94 / 0 | 1.00 / 83 / 0 | compensated; heavier -> slower (corr +0.58) |
| grasp_pos | 1.00 / 97 / 0 | 1.00 / 93 / 0 | 1.00 / 87 / 0 | |
| grasp_rot | 1.00 / 97 / 0 | 1.00 / 101 / 0 | 0.91 / 92 / 0 | |
| socket_pos | 1.00 / 93 / 0 | 1.00 / 91 / 0 | 1.00 / 89 / 0 | closed loop follows the moved socket |
| socket_rot | 1.00 / 93 / 0 | 0.97 / 92 / 0 | 0.81 / 76 / 0 | controller does not correct yaw/tilt |
| obs_socket_pos | 1.00 / 97 / 0 | **0.47 / 44 / 0** | **0.16 / 11 / 1** | aims at the biased observation (2.5 / 5 mm vs 3 mm success radius) |
| obs_eef_pos | 1.00 / 97 / 0 | **0.53 / 52 / 0** | **0.19 / 20 / 1** | same, via the gripper position |
| obs_eef_rot | 1.00 / 97 / 0 | 1.00 / 97 / 0 | 1.00 / 94 / 0 | not used by this controller |
| obs_socket_rot | 1.00 / 97 / 0 | 1.00 / 97 / 0 | 1.00 / 94 / 0 | not used by this controller |
| action_noise | 1.00 / 97 / 0 | 1.00 / 86 / 0 | 1.00 / 94 / 0 | |
| action_latency | 1.00 / 97 / 0 | 1.00 / 98 / 0 | 1.00 / 102 / 0 | |
| plug_wrench_force | 1.00 / 97 / 0 | 0.91 / 76 / 0 | 0.84 / 54 / 0 | |

## What this means for training
- The scripted controller is a weak stand-in for a policy: a policy can push harder (full actions), search
  with contact, and learn observation-bias correction. So low success here marks difficulty, not impossibility.
- Two cases are true physics hazards rather than difficulty: osc_stiffness and osc_damping_ratio at level 50
  eject plugs (4/32 each). Consider capping stiffness at ~1.5x and damping ratio at >= ~0.7x.
- joint_friction >= level 25 and mating_friction >= level 25 need much larger commands than 0.3; check that
  the policy's action range / reward allows it (joint friction at 0.7-0.8 N*m moved 58% at full action earlier).
- obs position bias of 2.5-5 mm is at or above the 3 mm success radius: the policy can only succeed if it learns
  to find the socket by contact.
- One ejection also happened at level 0 of plug_mass (1/32, a baseline config), so the unrandomized contact has
  a small blow-up rate too.
