# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify OSC payload gravity compensation (env.actions.arm_action.payload_gravity_compensation).

Runs identical-start tests with compensation OFF or ON (COMP=0/1) and prints one JSON line
("CHECK {...}") per test:
  hold    zero action for 150 steps (5 s): flange drift, plug slip in hand
  push    +x push (steps 5-15), -x push (45-55): x travel, z drift, slip
  insert  closed-loop scripted descent of the plug mate point onto the socket mate point:
          min / final mate distance, success fraction, steps to first success, slip, drops
With COMP=1 it also runs: hold at payload_mass_scale 2.0 (sign check: must RISE about as much as
OFF sinks) and 0.0 (must match OFF), and hold with per-env plug masses 0.5x..2x (must stay flat).

  source ../env.sh && $CONDA_RUN python payload_comp_check.py   (with PYTHONPATH set to the worktree)
"""

import argparse
import json
import os
import sys

import torch
import warp as wp

COMP = os.environ.get("COMP", "0") == "1"
VIDEO = os.environ.get("VIDEO", "0") == "1"  # record hold/push/rot/insert of env 0 (600 steps)
OUT = os.environ.get("OUT", os.path.join(os.environ.get("DR_TOOLS_RESULTS", "results"), "payload_comp"))
# Optional controller / joint overrides for gain sweeps:
#   STIFF="2000,2000,2000,300,300,300"  DAMP="0.7" (one value for all 6, or 6 values)  JFRIC="0.1" (N*m, all arm joints)
STIFF = os.environ.get("STIFF")
DAMP = os.environ.get("DAMP")
JFRIC = os.environ.get("JFRIC")
DECOUPLE = os.environ.get("DECOUPLE", "0") == "1"  # OSC inertial_dynamics_decoupling
PARTIAL = os.environ.get("PARTIAL", "0") == "1"  # partial_inertial_dynamics_decoupling
NULLSPACE = os.environ.get(
    "NULLSPACE"
)  # "stiffness:damping_ratio:target" e.g. "10:1.0:default" -> nullspace position control
TESTS = os.environ.get("TESTS", "hold,push,rot,insert").split(",")
INSERT_CLAMP = float(os.environ.get("INSERT_CLAMP", "0.3"))  # max |action| of the scripted insertion
LABEL = os.environ.get("LABEL", "")
sys.argv = sys.argv[:1]
import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import math as math_utils

from isaaclab_rl.entrypoints.common import apply_video_recording, enable_cameras_for_video, pre_launch_video_config

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

TASK = "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-ROS-Inference"
STEPS = 150
SCALE = 0.025  # metres per action unit


def t(x):
    return x.torch if hasattr(x, "torch") else wp.to_torch(x)


def main():  # noqa: C901 - one linear test script
    p = argparse.ArgumentParser()
    add_launcher_args(p)
    args = p.parse_args(["--viz", "kit" if VIDEO else "none"])
    if VIDEO:
        args.video = True
        args.video_length = 4 * STEPS
        args.video_interval = 10**9
        enable_cameras_for_video(args)
    overrides = ["env.actions.arm_action.payload_gravity_compensation=true"] if COMP else []
    env_cfg, _ = resolve_task_config(TASK, "", overrides=overrides)
    env_cfg.scene.num_envs = int(os.environ.get("N_ENVS", "8"))
    osc = env_cfg.actions.arm_action.controller_cfg
    if STIFF:
        osc.motion_stiffness_task = tuple(float(v) for v in STIFF.split(","))
    if DAMP:
        d = [float(v) for v in DAMP.split(",")]
        osc.motion_damping_ratio_task = tuple(d * 6 if len(d) == 1 else d)
    if DECOUPLE:
        osc.inertial_dynamics_decoupling = True
        osc.partial_inertial_dynamics_decoupling = PARTIAL
    if NULLSPACE:
        ns_k, ns_z, ns_target = NULLSPACE.split(":")
        osc.nullspace_control = "position"
        osc.nullspace_stiffness = float(ns_k)
        osc.nullspace_damping_ratio = float(ns_z)
        env_cfg.actions.arm_action.nullspace_joint_pos_target = ns_target
    if JFRIC:
        f = float(JFRIC)
        env_cfg.dr.enabled = True
        env_cfg.dr.joint_friction.enable = True
        env_cfg.dr.joint_friction.initial = (f, f)
        env_cfg.dr.joint_friction.final = (f, f)
    print(
        "CHECK "
        + json.dumps(
            {
                "test": "config",
                "label": LABEL,
                "comp": COMP,
                "stiffness": list(osc.motion_stiffness_task),
                "damping_ratio": [round(v, 3) for v in osc.motion_damping_ratio_task],
                "joint_friction": float(JFRIC) if JFRIC else 0.0,
                "decoupling": osc.inertial_dynamics_decoupling,
                "nullspace": NULLSPACE,
            }
        ),
        flush=True,
    )
    env_cfg.episode_length_s = 20.0
    env_cfg.terminations.plug_dropped = None
    env_cfg.terminations.plug_orientation_exceeded = None
    ev = env_cfg.events
    ev.randomize_socket_pose.params["pose_range"] = {k: [0.0, 0.0] for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    pc = ev.reset_plug_curriculum.params
    pc["at_goal_prob"] = 0.0
    pc["at_goal_prob_final"] = 0.0
    pc["normal_pose_range"] = {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]}
    pc["approach_depth_range"] = [0.04, 0.04]
    args.device = env_cfg.sim.device
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (0.75, 0.35, 0.25)
    env_cfg.viewer.lookat = (0.475, 0.125, 0.08)
    if VIDEO:
        pre_launch_video_config(env_cfg, args_cli=args)

    with launch_simulation(env_cfg, args):
        if VIDEO:
            apply_video_recording(env_cfg, OUT, args, subdir=f"comp{int(COMP)}")
        env = gym.make(TASK, cfg=env_cfg)
        u = env.unwrapped
        robot, plug, socket = u.scene["robot"], u.scene["dp_plug"], u.scene["dp_socket"]
        flange = robot.find_bodies(["flange"])[0][0]
        term = u.action_manager.get_term("arm_action")
        n, dev = u.num_envs, u.device
        root_q = t(robot.data.root_quat_w)

        def flange_pose():
            return t(robot.data.body_pos_w)[:, flange].clone(), t(robot.data.body_quat_w)[:, flange].clone()

        def plug_in_hand():
            fp, fq = flange_pose()
            rel = math_utils.quat_apply_inverse(fq, t(plug.data.root_pos_w) - fp)
            qrel = math_utils.quat_mul(math_utils.quat_inv(fq), t(plug.data.root_quat_w))
            return rel, qrel

        def mate_points():
            s_pos, s_quat = combine(socket, u._success_socket_offset, u._success_identity_quat)
            p_pos, _ = combine(plug, u._success_plug_offset, u._success_plug_goal_rot_inv)
            return s_pos, p_pos

        def combine(asset, off, quat_off):
            pos, quat = t(asset.data.root_pos_w), t(asset.data.root_quat_w)

            def batch(x):  # some env buffers are stored per env already
                return x if x.dim() == 2 else x.unsqueeze(0).expand(n, -1)

            return math_utils.combine_frame_transforms(pos, quat, batch(off), batch(quat_off))

        def slip(rel0, q0):
            rel1, q1 = plug_in_hand()
            dpos = torch.linalg.norm(rel1 - rel0, dim=-1) * 1000
            dang = torch.rad2deg(math_utils.quat_error_magnitude(q0, q1))
            return {"slip_mm_max": round(dpos.max().item(), 3), "slip_deg_max": round(dang.max().item(), 3)}

        marker = None
        if VIDEO:
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            marker = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/plug_start",
                    markers={
                        "start": sim_utils.SphereCfg(
                            radius=0.004, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.0))
                        )
                    },
                )
            )

        def mark_start():
            if marker is not None:
                marker.visualize(translations=t(plug.data.root_com_pos_w).clone())

        def action_zero():
            return torch.zeros(n, u.action_manager.total_action_dim, device=dev)

        def run_hold(label, per_env=False):
            env.reset()
            mark_start()
            f0, _ = flange_pose()
            rel0, q0 = plug_in_hand()
            for _ in range(STEPS):
                env.step(action_zero())
            f1, _ = flange_pose()
            d = (f1 - f0) * 1000
            out = {
                "test": label,
                "dx_mm": round(d[:, 0].mean().item(), 2),
                "dy_mm": round(d[:, 1].mean().item(), 2),
                "dz_mm": round(d[:, 2].mean().item(), 2),
                **slip(rel0, q0),
            }
            if per_env:
                out["dz_mm_per_env"] = [round(v, 2) for v in d[:, 2].tolist()]
            print("CHECK " + json.dumps(out), flush=True)

        def run_push():
            env.reset()
            mark_start()
            rel0, q0 = plug_in_hand()
            xs, zs = [], []
            for k in range(STEPS):
                a = action_zero()
                a[:, 0] = 0.3 if 5 <= k < 15 else (-0.3 if 45 <= k < 55 else 0.0)
                env.step(a)
                f, _ = flange_pose()
                xs.append(f[:, 0].mean().item())
                zs.append(f[:, 2].mean().item())
            out = {
                "test": "push",
                "push_dx_mm": round((xs[14] - xs[4]) * 1000, 2),
                "coast_after_push_mm": round((xs[44] - xs[14]) * 1000, 2),
                "return_dx_mm": round((xs[54] - xs[44]) * 1000, 2),
                "coast_after_return_mm": round((xs[-1] - xs[54]) * 1000, 2),
                "dz_total_mm": round((zs[-1] - zs[0]) * 1000, 2),
                **slip(rel0, q0),
            }
            print("CHECK " + json.dumps(out), flush=True)

        def run_rot():
            env.reset()
            mark_start()
            rel0, q0 = plug_in_hand()
            _, fq0 = flange_pose()
            f0, _ = flange_pose()
            ang = []
            for k in range(STEPS):
                a = action_zero()
                a[:, 5] = 0.3 if 5 <= k < 15 else (-0.3 if 45 <= k < 55 else 0.0)  # rotation about base z
                env.step(a)
                _, fq = flange_pose()
                ang.append(torch.rad2deg(math_utils.quat_error_magnitude(fq0, fq)).mean().item())
            f1, _ = flange_pose()
            out = {
                "test": "rot_yaw",
                "rot_after_push_deg": round(ang[14], 3),
                "rot_after_return_deg": round(ang[54], 3),
                "rot_end_deg": round(ang[-1], 3),
                "dz_total_mm": round((f1 - f0)[:, 2].mean().item() * 1000, 2),
                **slip(rel0, q0),
            }
            print("CHECK " + json.dumps(out), flush=True)

        def run_insert():
            env.reset()
            mark_start()
            rel0, q0 = plug_in_hand()
            d_min = torch.full((n,), 1e9, device=dev)
            first = torch.full((n,), -1, device=dev, dtype=torch.long)
            dropped = torch.zeros(n, dtype=torch.bool, device=dev)
            progress = {}
            diag = []  # [step, median mate dist mm, max |arm torque| N*m, max |arm qd| rad/s, max slip mm]
            peak_tau = torch.zeros(n, device=dev)
            seated_steps = torch.zeros(n, device=dev)
            axis_w = math_utils.quat_apply(
                t(socket.data.root_quat_w), torch.tensor([[1.0, 0.0, 0.0]], device=dev).expand(n, -1)
            )
            for k in range(STEPS):
                s_pos, p_pos = mate_points()
                err_b = math_utils.quat_apply_inverse(root_q, s_pos - p_pos)
                a = action_zero()
                a[:, 0:3] = torch.clamp(0.5 * err_b / SCALE, -INSERT_CLAMP, INSERT_CLAMP)
                env.step(a)
                s_pos, p_pos = mate_points()
                dist = torch.linalg.norm(p_pos - s_pos, dim=-1)
                d_min = torch.minimum(d_min, dist)
                succ = dist < u._success_pos_threshold
                first = torch.where((first < 0) & succ, torch.full_like(first, k), first)
                rel_now, _ = plug_in_hand()
                dropped |= torch.linalg.norm(rel_now - rel0, dim=-1) > 0.02  # plug left the grasp
                if k in (10, 20, 30, 40, 60, 80, 120):
                    progress[k] = round(dist.mean().item() * 1000, 2)
                peak_tau = torch.maximum(peak_tau, term._joint_efforts.abs().max(dim=-1).values)
                seated_steps += (dist < u._success_pos_threshold).float()
                if k % 10 == 9:
                    qd = t(robot.data.joint_vel)[:, term._joint_ids]
                    diag.append(
                        [
                            k + 1,
                            round(dist.median().item() * 1000, 2),
                            round(term._joint_efforts.abs().max().item(), 2),
                            round(qd.abs().max().item(), 3),
                            round((torch.linalg.norm(rel_now - rel0, dim=-1).max() * 1000).item(), 2),
                        ]
                    )
            s_pos, p_pos = mate_points()
            final = torch.linalg.norm(p_pos - s_pos, dim=-1) * 1000
            ok = first >= 0
            out = {
                "test": "insert",
                "min_mate_dist_mm": round(d_min.mean().item() * 1000, 2),
                "final_mate_dist_mm": round(final.mean().item(), 2),
                "success_frac": round(ok.float().mean().item(), 3),
                "first_success_step_mean": round(first[ok].float().mean().item(), 1) if ok.any() else None,
                "dropped": int(dropped.sum()),
                "mate_dist_mm_at_step": progress,
                "insertion_axis_w": [round(v, 3) for v in axis_w[0].tolist()],
                "diag": diag,
                "blowup_envs": int((peak_tau > 500).sum()),
                "peak_tau_median": round(peak_tau.median().item(), 2),
                "seated_steps_mean": round(seated_steps.mean().item(), 1),
                **slip(rel0, q0),
            }
            print("CHECK " + json.dumps(out), flush=True)

        if "hold" in TESTS:
            run_hold("hold")
        if "push" in TESTS:
            run_push()
        if "rot" in TESTS:
            run_rot()
        if "insert" in TESTS:
            run_insert()
        if COMP:
            term.cfg.payload_mass_scale = 2.0
            run_hold("hold_scale2.0")
            term.cfg.payload_mass_scale = 0.0
            run_hold("hold_scale0.0")
            term.cfg.payload_mass_scale = 1.0
            base = t(plug.data.body_mass).clone()
            factors = torch.linspace(0.5, 2.0, n, device=dev).unsqueeze(-1)
            plug.set_masses_index(
                masses=(base * factors).cpu() if base.device.type == "cpu" else base * factors,
                env_ids=torch.arange(n, device=dev),
            )
            print(
                "CHECK "
                + json.dumps(
                    {
                        "test": "mass_readback",
                        "plug_mass_g": [round(v * 1000, 1) for v in t(plug.data.body_mass)[:, 0].tolist()],
                    }
                ),
                flush=True,
            )
            run_hold("hold_mass0.5to2x", per_env=True)
        print("CHECK_DONE", flush=True)
        env.close()


if __name__ == "__main__":
    main()
