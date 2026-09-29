# Payload gravity compensation check (2026-09-26)

Harness: `dev/payload_comp_check.py` (8 envs, identical starts), videos via `dev/payload_comp_videos.sh`.
Compensation: `env.actions.arm_action.payload_gravity_compensation=true` (commit b221791bbbf).

## Why it is needed
The task-space arm is gravity-free, but the plug is not. The OSC's `pose_rel` target is re-based on the measured
pose every step, so the plug's weight (30 g) is never corrected: under zero actions the gripper sinks steadily.
The real robot is a stiff 1 kHz Cartesian position servo and does not sag. ROS sets no payload.

## Results (compensation OFF -> ON)
| test | OFF | ON |
|---|---|---|
| hold, zero action 5 s: dz | -28.8 mm | +0.05 mm |
| push +x 0.3 then return: travel / dz | 9.55 mm / -31.0 mm | 9.76 mm / +0.44 mm |
| yaw 0.3: reached / after return | 2.66 / 0.10 deg | 2.66 / 0.05 deg |
| scripted insertion | 8/8, 0 dropped | 8/8, 0 dropped |
| `payload_mass_scale=2.0` (over-compensation) | - | rises +32.3 mm (sign check) |
| `payload_mass_scale=0.0` | - | sinks -28.6 mm (same as OFF) |
| per-env plug mass 15-60 g (plug_mass DR x0.5-2) | - | dz -0.17 to +0.12 mm per env |

Compensation reads each env's current plug mass, so it follows the plug_mass DR. Free motion, rotation and
contact are unchanged. It adds `J^T [m*g_b ; lever x m*g_b]` (lever = plug CoM relative to the EE frame) to
the OSC joint efforts every physics step.
