# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Score a knob sweep's results.jsonl: is each knob applied correctly, and does it have its intended effect?

Usage: python3 dr_knob_sweep_report.py [results.jsonl]   -> prints a markdown report
"""

import json
import os
import sys

PATH = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.path.join(os.environ.get("DR_TOOLS_RESULTS", "results"), "knob_sweep/results.jsonl")
)

RANGE_KEY = {
    "osc_stiffness": "scale_range",
    "osc_damping_ratio": "scale_range",
    "joint_armature": "value_range",
    "joint_friction": "value_range",
    "finger_friction": "finger_static_mu_range",
    "mating_friction": "plug_static_mu_range",
    "plug_mass": "mass_scale_range",
    "plug_wrench_force": "force_component_range_N",
}
AXIS_KEY = {
    "grasp_pos": "injected_max_abs_mm_xyz",
    "grasp_rot": "injected_max_abs_deg_rpy",
    "socket_pos": "offset_max_abs_mm_xyz",
    "socket_rot": "offset_max_abs_deg_rpy",
}
NOISE_KNOBS = {"obs_socket_pos", "obs_eef_pos", "obs_eef_rot", "obs_socket_rot", "action_noise"}
# Expected sign of the per-env correlation between the sampled value and the response.
SIGN = {
    "osc_stiffness": ("corr_param_vs_push_disp", +1, "stiffer -> moves further"),
    "osc_damping_ratio": ("corr_param_vs_push_disp", -1, "more damping -> moves less"),
    "joint_armature": ("corr_param_vs_push_disp", -1, "more armature -> moves less"),
    "joint_friction": ("corr_param_vs_push_disp", -1, "more friction -> moves less"),
    "plug_mass": ("corr_mass_vs_z_drop", +1, "heavier -> sags more"),
    "action_latency": ("corr_latency_vs_onset", +1, "more latency -> starts later"),
}
# Knobs that change the arm's physical response to an identical command.
DYNAMIC = {
    "osc_stiffness",
    "osc_damping_ratio",
    "joint_armature",
    "joint_friction",
    "plug_mass",
    "action_noise",
    "action_latency",
    "plug_wrench_force",
}
# Knobs that should NOT change the physics (only what the policy sees).
NO_PHYSICS = {"obs_socket_pos", "obs_eef_pos", "obs_eef_rot", "obs_socket_rot"}


def check_range(exp, meas):
    lo, hi = (exp[0], exp[1]) if not isinstance(exp[0], list) else exp
    if isinstance(lo, list):
        return None, "n/a"
    width = hi - lo
    if lo > hi:  # symmetric wrench ranges are written (-h, h)
        lo, hi, width = hi, lo, lo - hi
    tol = 0.02 * max(abs(lo), abs(hi), 1e-3) + 1e-4
    inside = meas[0] >= lo - tol and meas[1] <= hi + tol
    if width < 1e-9:
        ok = inside and abs(meas[0] - lo) <= tol and abs(meas[1] - hi) <= tol
        return ok, f"{meas[0]:.4g}..{meas[1]:.4g} vs {lo:.4g}"
    cover = (meas[1] - meas[0]) / width
    return inside and cover >= 0.6, f"{meas[0]:.4g}..{meas[1]:.4g} vs {lo:.4g}..{hi:.4g} ({cover:.0%} covered)"


def check_axis(exp, meas):
    ok = True
    for e, m in zip(exp, meas):
        if e <= 1e-9:
            ok &= m <= 0.3
        else:
            ok &= 0.6 * e <= m <= 1.1 * e + 0.2
    return ok, f"{meas} vs +/-{exp}"


def check_noise(exp, meas):
    bh, nh = exp["bias_halfwidth"], exp["noise_halfwidth"]
    # For actions the per-episode bias is read straight off the action term, which is exact;
    # the 30-sample estimate is swamped when the per-step noise is larger than the bias.
    b = meas.get("term_bias_max_abs", meas["bias_max_abs"])
    n = meas["noise_max_abs"]
    eps = 1e-5
    ok_b = (b <= eps) if bh <= eps else (0.6 * bh <= b <= 1.15 * bh + eps)
    # The per-step estimate is sample minus its own 30-sample mean, so it can legitimately
    # exceed the half-width by ~30%; the bias estimate is far tighter.
    ok_n = (n <= eps) if nh <= eps else (0.5 * nh <= n <= 1.4 * nh + eps)
    return ok_b and ok_n, f"bias {b:.4g} vs {bh:.4g}, noise {n:.4g} vs {nh:.4g}"


def check_latency(exp, seen):
    lo, hi = exp
    allowed = {int(round(lo + (hi - lo) * i / 100)) for i in range(101)}
    return set(seen) <= allowed, f"seen {seen} vs range {lo:g}..{hi:g}"


def main():
    with open(PATH) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    # Knobs removed from the config since the sweep was recorded.
    rows = [r for r in rows if r["knob"] not in {"plug_wrench_torque"}]
    print("| knob | level | applied | detail | spread mm | effect |")
    print("|---|---|---|---|---|---|")
    verdicts = {}
    for r in rows:
        k = r["knob"]
        lv = r["levels"]
        spreads = [lv[L]["effect"]["response_spread_mm"] for L in ("0", "25", "50")]
        all_applied = True
        for L in ("0", "25", "50"):
            ph = lv[L]
            exp, m = ph["expected"], ph["measured"]
            if k in RANGE_KEY:
                ok, det = check_range(exp, m[RANGE_KEY[k]])
                if k == "plug_wrench_force" and L != "0":
                    ok = ok and m.get("force_has_both_signs", True)
            elif k in AXIS_KEY:
                ok, det = check_axis(exp, m[AXIS_KEY[k]])
            elif k in NOISE_KNOBS:
                ok, det = check_noise(exp, m)
            elif k == "action_latency":
                ok, det = check_latency(exp, m["latency_steps_seen"])
            else:
                ok, det = None, "no check"
            all_applied &= bool(ok)
            eff = ""
            if k in SIGN and L != "0":
                key, sign, why = SIGN[k]
                c = ph["effect"].get(key)
                good = c is not None and c * sign > 0.3
                eff = f"corr {c} ({why}) {'OK' if good else 'WEAK/WRONG'}"
            elif k == "finger_friction" and L != "0":
                eff = f"corr {ph['effect'].get('corr_param_vs_push_disp')} (expected ~0: grip is geometric)"
            print(f"| {k} | {L} | {'PASS' if ok else 'FAIL'} | {det} | {ph['effect']['response_spread_mm']} | {eff} |")
        if k in DYNAMIC:
            grows = spreads[0] < spreads[1] < spreads[2]
            # Band-type ranges (e.g. friction 4-5) move every env together, which shows up as a
            # shift in the mean response rather than as spread between envs.
            mids = [sum(lv[L]["effect"]["push_disp_mm_range"]) / 2 for L in ("0", "25", "50")]
            shift = [round(100 * (m - mids[0]) / mids[0], 1) for m in mids]
            shifted = abs(shift[2]) >= 3.0
            tag = "grows" if grows else "flat"
            effect = (
                f"spread {spreads[0]}->{spreads[1]}->{spreads[2]} mm ({tag}); "
                f"mean push shift 0/{shift[1]}/{shift[2]}% -> {'EFFECT' if (grows or shifted) else 'NO EFFECT'}"
            )
        elif k in NO_PHYSICS:
            flat = max(spreads) < 1.0
            effect = f"physics unchanged as expected ({spreads} mm)" if flat else f"UNEXPECTED physics change {spreads}"
        else:
            effect = f"spread {spreads} mm"
        verdicts[k] = (all_applied, effect)

    print("\n| knob | applied at 0/25/50 | effect |")
    print("|---|---|---|")
    for k, (ok, effect) in verdicts.items():
        print(f"| {k} | {'PASS' if ok else 'FAIL'} | {effect} |")


if __name__ == "__main__":
    main()
