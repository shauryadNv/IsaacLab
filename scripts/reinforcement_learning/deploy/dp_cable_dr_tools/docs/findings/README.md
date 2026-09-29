# Findings

Write-ups of the local experiments run with `dev/`. Raw outputs (jsonl, logs, videos) were written to
`$DR_TOOLS_RESULTS/<experiment>/` and are not committed (videos are large). HANDOFF.md summarizes all of them.

| File | Experiment | Script |
|---|---|---|
| `payload_comp_SUMMARY.md` | payload gravity compensation off vs on | `dev/payload_comp_check.py`, `dev/payload_comp_videos.sh` |
| `gain_sweep_SUMMARY.md` | can OSC gains / joint friction fix the sag instead (no) | `dev/gain_sweep.sh` |
| `decoupled_search_SUMMARY.md` | inertial-decoupled OSC gain search for stable contact | `dev/decoupled_search.sh`, `dev/decoupled_followup.sh`, `dev/decoupled_rot_search.sh` |
| `knob_sweep_comp_insert_REPORT.md` | every DR knob at ADR 0/25/50 with compensation + scripted insertion | `dev/dr_knob_sweep.sh` (`SWEEP_COMP=1 SWEEP_EXCITE=insert`) |
| `osc_cap_verify_SUMMARY.md` | which OSC gain DR caps keep contact stable (softer / more-damped only) | `dev/osc_cap_verify.sh`, `dev/osc_cap_diag*.sh` |
| `std_instability_REPORT.md` | training instability: resume-LR bug + per-state std blow-up | osmo sweeps + `dev/std_probe.py` |
