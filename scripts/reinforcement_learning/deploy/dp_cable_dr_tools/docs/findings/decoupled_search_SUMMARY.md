# Decoupled-OSC gain search for stable socket contact (2026-09-26)

All runs: `inertial_dynamics_decoupling=True`, payload gravity compensation ON, 8 envs, identical starts.
Harness `dev/payload_comp_check.py`; scripts `decoupled_search.sh` (phase 1+2), `decoupled_followup.sh`.
Blow-up = any arm joint torque > 500 N*m during insertion (the articulation diverges while the plug is seated;
the plug stays in the socket and the arm flies off). Hard insertion = scripted insertion allowed full-scale
actions (clamp 1.0) instead of 0.3.

## Stability boundary (phase 1, insertion)
- Translational K <= 200 (1/s^2): 0 blow-ups. K 300: 2-4 of 8 blow up; K 500: 8/8; K 2000/300: 8/8.
- Partial decoupling and nullspace position control (10 / 1.0 to default pose) do NOT move the boundary.

## Stable candidates (phase 2 + follow-up)
| config (trans / rot K, zeta) | hold dz | push / coast | yaw reached | insert 0.3 | insert 1.0 |
|---|---|---|---|---|---|
| 100/20, 1.0 | +0.09 | 10.1 / 1.5 mm | 0.22 deg | 8/8, 0 drop | 8/8, 0 drop |
| 100/20, 0.7 | +0.04 | 13.2 / 3.1 | 0.26 | 8/8 | 8/8 |
| 200/30, 1.0 | +0.06 | 14.6 / 1.4 | 0.29 | 8/8 | 8/8 |
| 200/30, 0.7 | +0.07 | 19.3 / 2.8 | 0.35 | 8/8 | 2 drops |
| 100/50, 1.0 | +0.10 | 10.0 / 1.5 | 0.38 | 8/8 | 8/8 |
| **100/75, 1.0** | +0.10 | 10.0 / 1.5 | 0.49 | 8/8 | 8/8 |
| 100/100, 1.0 | +0.10 | 10.0 / 1.5 | 0.59 | 8/8 | 8/8 |
| current (no decoupling, 300/30) + comp | +0.05 | 9.8 / 8.7 | 2.66 | 8/8 | - |

DR-extreme stress (osc_stiffness x2, damping ratio 0.5): 200/100 and 200/150 at zeta 0.5 blow up in 3-4 of 8
hard insertions -> the current osc gain DR ranges (stiffness 0.5-2x, damping ratio 0.5-1.5x) are unsafe with
decoupling; they would need narrowing (e.g. stiffness <= 1.5x, damping ratio >= 0.7).

## Takeaways
- Decoupling CAN be contact-stable, but only with soft gains (translational K ~100-200).
- At those gains it removes coasting (8.7 -> 1.5 mm) with the same push travel, but rotation authority drops
  ~4-10x (0.2-0.6 deg vs 2.66 deg for the same yaw action).
- Neither mode tracks per-step deltas like a stiff servo would: a 0.3 action for 10 steps asks for 75 mm and
  the sim moves ~10 mm in both modes.
- Best decoupled candidate: 100/100/100/75/75/75, zeta 1.0, + payload compensation, with narrowed osc DR.

## Follow-up 2 (2026-09-27): stiff rotation + no-compensation check (`decoupled_rot_search.sh`)
Rotation was too SOFT in decoupled mode (unit-mass dynamics: K 75 -> 8.7 rad/s, slower than a 33 ms step).
Raising rotational K with translation at 100, zeta 1.0, compensation ON:
| rot K | yaw reached | returns to | push / coast | insert 0.3 / 1.0 |
|---|---|---|---|---|
| 300 | 1.03 deg | 0.15 | 10.0 / 1.5 | 8/8 / 8/8 |
| 1000 | 1.78 | 0.13 | 10.0 / 1.5 | 8/8 / 8/8 |
| **3000** | **2.75** | **0.08** | 10.0 / 1.5 | 8/8 / 8/8, 0 drops, 0 blow-ups |
Best: decoupled 100/100/100/3000/3000/3000, zeta 1.0 + payload comp = current push and yaw authority,
no coasting, stable contact (insertion somewhat slower). Not yet stress-tested at osc DR extremes.
Without payload compensation: decoupled 100/75 sinks -11.9 mm, 100/1000 sinks -7.6 mm in 5 s -> still needed.

## DR-extreme stress for decoupled 100/3000, zeta 1.0 + comp (osc_stiffness / osc_damping_ratio at ADR 50)
| extreme | hold dz | push / coast | yaw | insert 0.3 | insert 1.0 |
|---|---|---|---|---|---|
| stiffness x2, damping x0.5 (200/6000, zeta 0.5) | +0.03 | 24.4 / 5.0 | 5.0 deg | 7/8, 1 blow-up | 8/8, **3 blow-ups** |
| stiffness x0.5, damping x1.5 (50/1500, zeta 1.5) | +0.09 | 5.0 / 0.8 | 1.5 deg | 4/8, 0 blow-ups | 4/8, 0 blow-ups |
-> same as the current controller: the high-stiffness / low-damping corner of the osc gain DR is not contact-safe.
If this controller is adopted, cap the osc DR (e.g. stiffness <= 1.5x, damping ratio >= 0.7x).
