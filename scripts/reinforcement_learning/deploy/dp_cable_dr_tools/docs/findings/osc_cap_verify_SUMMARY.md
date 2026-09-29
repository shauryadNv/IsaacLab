# OSC gain DR caps (2026-09-27/28)

Goal: stop osc_stiffness / osc_damping_ratio randomization from making insertion contact blow up (the knob
sweep had 4/32 plugs ejected at ADR 50 with the original 0.5-2x / 0.5-1.5x ranges).
Harness: `dev/dr_knob_sweep.sh` (SWEEP_EXCITE=insert) and `payload_comp_check.py` corners
(`osc_cap_verify.sh`, `osc_cap_diag.sh`, `osc_cap_diag2.sh`), 32 envs, scripted insertion.

## 1. Requested caps (stiffness <= 1.5x, damping ratio >= 0.7x) do NOT fix it
- Knob sweep L50: stiffness 3-5/32 ejected (comp off, or full actions), damping 4-5/32.
- Worst corner (1.5x stiffness AND 0.7x damping on every axis): 10-31 of 32 envs blow up.

## 2. Which axis? (corners, comp ON, full-scale actions)
| corner | success | blow-ups |
|---|---|---|
| nominal | 100% | 0 |
| rotation only 1.5x / 0.7x | 100% | 0 |
| translation only 1.5x / 0.7x | 22% | 26 |
| both 1.25x / 0.85x | 9% | 14 |
| both 1.2x / 0.9x | 44% | 12 |
| translation stiffness 1.1x | 84% (comp off 75%) | 2 (comp off 24) |
| translation damping 0.9x | 100% (comp off 19%) | 10 (comp off 15) |
| stiffness 0.5x | 100% (comp off 94%) | 0 |
| damping 1.5x | 88% (comp off 100%) | 0 |
| stiffness 0.5x + damping 1.5x | 94% (comp off 31%) | 0 |
-> the nominal translational gains have no margin: stiffer or less damped translation blows up; softer
   or more damped is stable. Rotation randomization is harmless.

## 3. Adopted: softer-only / more-damped-only (commit c93192d3619)
osc_stiffness final (0.5, 1.0), osc_damping_ratio final (1.0, 1.5), log-uniform. Realistic knob sweep, L50:
| setting | L0 (nominal) | stiffness 0.5-1.0 | damping 1.0-1.5 |
|---|---|---|---|
| comp off, clamp 0.3 | 100%, 0 ejected | 88%, 0 | 100%, 0 |
| comp on, full actions | 94%, 0 | 63%, 0 | 88%, 1 |
| comp off, full actions | 78%, 21 ejected | 84%, 5 | 94%, 5 |
The capped DR never adds ejections beyond nominal. Remaining issue (not DR): with compensation OFF and
full-scale pushing, even the nominal controller ejects plugs (21/32); with compensation ON it does not.

Launched: 16 runs (osc_stiffness, osc_damping_ratio, grp_controller, all_on) x 4 curriculum/comp combos,
sweeps `*_osccap`, runs 1320-1335 (all_on at HIGH).
