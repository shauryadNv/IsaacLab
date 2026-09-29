# Can controller gains / joint friction fix the plug-weight sag? (2026-09-26)

Harness: `dev/payload_comp_check.py` (8 envs, identical starts, payload compensation OFF).
Tests: hold (zero action 5 s), push (+x/-x 0.3), rotate (+/-yaw 0.3), scripted insertion (P-control of the
plug mate point onto the socket mate point, 150 steps). Raw rows: `results*.jsonl`, per-config logs here.
"Ejected" = plug left the grasp during insertion (the huge mean distances in the raw rows come from these).

## Baseline and the fix that works
| config | hold dz | push travel / coast | insertion |
|---|---|---|---|
| current (K 300/30 force-level, ratio 1.01/0.10, no decoupling) | -28.8 mm | 9.6 / 8.7 mm | 8/8, 0 ejected |
| + payload gravity compensation (`payload_comp/`) | +0.05 mm | 9.8 / 8.7 mm | 8/8, 0 ejected |

## Without inertial decoupling (gains are raw forces)
- ratio 0.7 on ALL axes (K 300/30, 500/75, 2000/300): unstable in free space (rotational D = 2*0.7*sqrt(K)
  is too large for the light wrist at 240 Hz); plugs flung.
- stiffer translation only, rotation kept at 30 / 0.10: sag 29 -> 25 (K 500) -> 11-14 mm (K 2000), coasting
  unchanged (9-14 mm), insertion ejects 1-8 of 8 plugs (stiff force-level gains make socket contact violent).
- joint friction 0.05 / 0.1 / 0.2 N*m: sag -18 / -7 / -0.5 mm, but small commands are blocked; insertion
  4/8, 0/8, 0/8 (stalls ~10 mm short). Rotation nearly frozen at 0.2.

## With inertial decoupling (F = Lambda(q) * a; gains become accelerations, zeta a true damping ratio)
- sag -2 to -8 mm, coasting ~0-2 mm (tracks like a servo), same action moves 2-5x further (17-53 mm).
- current rotational ratio 0.10 is genuinely underdamped here (yaw ends 5-11 deg off); zeta 0.7 fixes it.
- insertion: reaches the goal fast (steps 10-36) but then ALL 8 plugs are ejected in every decoupled config
  (Lambda * K makes contact forces far larger than the SDF contact can take). Would need retuning (softer
  gains / contact settings) before it is usable for training.

## Takeaway
Payload gravity compensation is the only option that removes the sag with no side effect on free motion,
contact, or grasp. Decoupling is the route to also remove coasting (closer to the real servo) but needs a
contact-stable gain search first; it also changes action authority, so policies would need retraining.
