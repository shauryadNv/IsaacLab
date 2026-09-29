# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-knob verification of the DP domain randomization at ADR level 0, 25 and 50.

For ONE knob (only that knob enabled), switches the ADR level at runtime 0 -> 25 -> 50,
resetting after each switch, which exercises the same curriculum path training uses.

Each phase checks two things:
  * applied  -- the values the simulator actually used, against the range expected at
                that level: initial + (level/50) * (final - initial)
  * effect   -- all envs start identically and run the same scripted push-and-return, so
                any spread in their response comes from the knob. Dynamics knobs also get a
                sign check (e.g. stiffer -> moves further, more latency -> starts later).

Records one close-up video of env 0 across the three phases; for position observation noise,
green/red spheres mark the true vs observed position.

Env vars: SWEEP_KNOB (required), SWEEP_OUT, SWEEP_ENVS (32), SWEEP_PHASE_STEPS (150),
          SWEEP_LEVELS ("0,25,50"), SWEEP_FINAL ("lo,hi" overrides a scalar knob's final range),
          SWEEP_VIDEO ("1"; "0" runs headless without recording),
          SWEEP_COMP ("1" = payload gravity compensation ON in the OSC action),
          SWEEP_EXCITE ("push" = +x/-x push-and-return; "insert" = scripted insertion of the plug mate
          point onto the socket mate point, |action| <= SWEEP_PUSH; for obs_socket_pos / obs_eef_pos the
          controller steers on the noisy observed position, as a policy would)
"""

import argparse
import json
import math
import os
import sys

import gymnasium as gym
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import math as math_utils

from isaaclab_rl.entrypoints.common import apply_video_recording, enable_cameras_for_video, pre_launch_video_config

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

TASK = "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-ROS-Inference"
LEVELS = [0, 25, 50]
PUSH = float(os.environ.get("SWEEP_PUSH", "0.3"))  # action units; 0.3 -> target leads TCP by 7.5 mm/step
OBS_TERM = {
    "obs_socket_pos": "socket_kp_pos",
    "obs_eef_pos": "eef_pos",
    "obs_eef_rot": "eef_rot_6d",
    "obs_socket_rot": "socket_kp_rot_6d",
}


def t(x):
    return x.torch if hasattr(x, "torch") else wp.to_torch(x)


def action_x(k):
    """Scripted excitation: push +x, hold, push back -x, hold."""
    if 5 <= k < 15:
        return PUSH
    if 45 <= k < 55:
        return -PUSH
    return 0.0


def lerp(a, b, f):
    if isinstance(a, (tuple, list)):
        return [lerp(x, y, f) for x, y in zip(a, b)]
    return a + f * (b - a)


def corr(x, y):
    x, y = x.float().flatten(), y.float().flatten()
    if x.std() < 1e-12 or y.std() < 1e-12:
        return None
    return round(torch.corrcoef(torch.stack([x, y]))[0, 1].item(), 3)


def rng(v):
    return [round(v.min().item(), 5), round(v.max().item(), 5)]


def main():  # noqa: C901 - one linear test script
    knob = os.environ["SWEEP_KNOB"]
    n_envs = int(os.environ.get("SWEEP_ENVS", "32"))
    steps = int(os.environ.get("SWEEP_PHASE_STEPS", "150"))
    out_root = os.environ.get("SWEEP_OUT", os.path.join(os.environ.get("DR_TOOLS_RESULTS", "results"), "knob_sweep"))
    levels = [int(v) for v in os.environ.get("SWEEP_LEVELS", "0,25,50").split(",")]
    final_override = os.environ.get("SWEEP_FINAL")
    video = os.environ.get("SWEEP_VIDEO", "1") == "1"
    comp = os.environ.get("SWEEP_COMP", "0") == "1"
    excite = os.environ.get("SWEEP_EXCITE", "push")
    sys.argv = sys.argv[:1]

    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    args = parser.parse_args(["--viz", "kit" if video else "none"])
    args.video = video
    args.video_length = len(levels) * steps
    args.video_interval = 10**9
    enable_cameras_for_video(args)

    torch.manual_seed(0)
    env_cfg, _ = resolve_task_config(TASK, "")
    env_cfg.scene.num_envs = n_envs
    env_cfg.episode_length_s = steps / 30.0 + 5.0
    # Only time_out may end an episode, so no env resets mid-phase.
    env_cfg.terminations.plug_dropped = None
    env_cfg.terminations.plug_orientation_exceeded = None
    # Identical starting states: remove the env's own socket/plug jitter. (The socket knobs
    # replace the socket jitter with their own range when enabled.)
    ev = env_cfg.events
    ev.randomize_socket_pose.params["pose_range"] = {k: [0.0, 0.0] for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    pc = ev.reset_plug_curriculum.params
    pc["at_goal_prob"] = 0.0
    pc["at_goal_prob_final"] = 0.0
    pc["normal_pose_range"] = {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]}
    pc["approach_depth_range"] = [0.04, 0.04]

    if comp:
        env_cfg.actions.arm_action.payload_gravity_compensation = True

    dr = env_cfg.dr
    dr.enabled = True
    getattr(dr, knob).enable = True
    if knob == "mating_friction":
        dr.mating_friction.final = (0.001, 0.5)  # the default is identity; give it a range to test
    if final_override:
        getattr(dr, knob).final = tuple(float(v) for v in final_override.split(","))
    dr.adr.enable = True
    dr.adr.init_level = 0
    dr.adr.success_threshold = 99.0  # the level only changes when this script sets it

    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (0.75, 0.35, 0.25)
    env_cfg.viewer.lookat = (0.475, 0.125, 0.08)
    args.device = env_cfg.sim.device
    if video:
        pre_launch_video_config(env_cfg, args_cli=args)

    kcfg = getattr(dr, knob)
    result = {
        "knob": knob,
        "n_envs": n_envs,
        "comp": comp,
        "excite": excite,
        "final": list(getattr(dr, knob).final) if hasattr(getattr(dr, knob), "final") else None,
        "levels": {},
    }

    with launch_simulation(env_cfg, args):
        if video:
            apply_video_recording(env_cfg, out_root, args, subdir=knob)
        env = gym.make(TASK, cfg=env_cfg)
        u = env.unwrapped
        dev = u.device
        robot, plug, socket = u.scene["robot"], u.scene["dp_plug"], u.scene["dp_socket"]
        eef = robot.find_bodies(["flange"])[0][0]
        arm = robot.find_joints(["joint[1-7]"])[0]
        term = u.action_manager.get_term("arm_action")
        sched = u.curriculum_manager.cfg["adr"].func
        origin0 = u.scene.env_origins[0]

        om = u.observation_manager
        names = om.active_terms["policy"]
        dims = [d[0] for d in om.group_obs_term_dim["policy"]]
        cfgs = om._group_obs_term_cfgs["policy"]
        offs = {n: (sum(dims[:i]), sum(dims[: i + 1])) for i, n in enumerate(names)}

        markers = None
        if knob in ("obs_socket_pos", "obs_eef_pos"):
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            markers = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/dr_obs",
                    markers={
                        "true": sim_utils.SphereCfg(
                            radius=0.003, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.9, 0.0))
                        ),
                        "observed": sim_utils.SphereCfg(
                            radius=0.003, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.0, 0.0))
                        ),
                    },
                )
            )

        root_q = t(robot.data.root_quat_w)

        def mate_points():
            def frame(asset, off, qoff):
                n_ = u.num_envs
                b = lambda x: x if x.dim() == 2 else x.unsqueeze(0).expand(n_, -1)  # noqa: E731
                return math_utils.combine_frame_transforms(
                    t(asset.data.root_pos_w), t(asset.data.root_quat_w), b(off), b(qoff)
                )[0]

            return (
                frame(socket, u._success_socket_offset, u._success_identity_quat),
                frame(plug, u._success_plug_offset, u._success_plug_goal_rot_inv),
            )

        def plug_rel():
            fpos = t(robot.data.body_pos_w)[:, eef]
            fquat = t(robot.data.body_quat_w)[:, eef]
            rel = math_utils.quat_apply_inverse(fquat, t(plug.data.root_pos_w) - fpos)
            qrel = math_utils.quat_mul(math_utils.quat_inv(fquat), t(plug.data.root_quat_w))
            return rel, qrel

        nominal = {}
        finger_shapes = None

        for level in levels:
            f = level / 50.0
            sched.level = level
            env.reset()  # curriculum interpolates to `level`, then reset events resample

            # Before any physics step: exactly what the reset events wrote.
            rel0, qrel0 = plug_rel()
            zero = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=dev)
            env.step(zero)
            rel1, qrel1 = plug_rel()
            sock_pos = t(socket.data.root_pos_w) - u.scene.env_origins
            sock_quat = t(socket.data.root_quat_w)
            if level == 0:
                # Grasp is unrandomized at level 0, so env 0 is a valid reference. The socket is
                # randomized even at level 0, so its reference must be the default pose instead.
                nominal["rel0"] = rel0[0].clone()
                nominal["q0"] = qrel0[0].clone()
                nominal["rel"] = rel1[0].clone()
                nominal["q"] = qrel1[0].clone()
                nominal["mass"] = t(plug.data.body_mass)[0, 0].item()
            default_pose = t(socket.data.default_root_pose)
            nominal["sock_pos"] = default_pose[:, :3]
            nominal["sock_quat"] = default_pose[:, 3:7]

            ph = {"expected": None, "measured": {}}
            m = ph["measured"]

            # ---------------- applied values ----------------
            if knob in ("osc_stiffness", "osc_damping_ratio"):
                osc = term._osc
                p = torch.diagonal(osc._motion_p_gains_task, dim1=-2, dim2=-1)
                d = torch.diagonal(osc._motion_d_gains_task, dim1=-2, dim2=-1)
                kp_nom = torch.tensor(osc.cfg.motion_stiffness_task, device=dev)
                zeta_nom = torch.tensor(osc.cfg.motion_damping_ratio_task, device=dev)
                if knob == "osc_stiffness":
                    vals = p / kp_nom
                else:
                    vals = d / (2.0 * p.sqrt()) / zeta_nom
                m["scale_range"] = rng(vals)
                m["scale_median"] = round(vals.median().item(), 3)
                ph["expected"] = lerp(kcfg.initial, kcfg.final, f)
                per_env_param = vals[:, :3].mean(-1)
            elif knob in ("joint_armature", "joint_friction"):
                data = robot.data.joint_armature if knob == "joint_armature" else robot.data.joint_friction_coeff
                vals = t(data)[:, arm]
                m["value_range"] = rng(vals)
                ph["expected"] = lerp(kcfg.initial, kcfg.final, f)
                per_env_param = vals.mean(-1)
            elif knob == "finger_friction":
                mats = wp.to_torch(robot.root_view.get_material_properties())[..., 0]
                if finger_shapes is None:
                    finger_shapes = (mats[0] - kcfg.initial[0]).abs() < 1e-3
                vals = mats[:, finger_shapes]
                m["finger_static_mu_range"] = rng(vals)
                m["n_finger_shapes"] = int(finger_shapes.sum().item())
                ph["expected"] = lerp(kcfg.initial, kcfg.final, f)
                per_env_param = vals.mean(-1).to(dev)
            elif knob == "mating_friction":
                vals = wp.to_torch(plug.root_view.get_material_properties())[..., 0]
                m["plug_static_mu_range"] = rng(vals)
                ph["expected"] = lerp(kcfg.initial, kcfg.final, f)
            elif knob == "plug_mass":
                vals = t(plug.data.body_mass)[:, 0] / nominal["mass"]
                m["mass_scale_range"] = rng(vals)
                ph["expected"] = lerp(kcfg.initial, kcfg.final, f)
                per_env_param = vals
            elif knob == "grasp_pos":
                inj = (rel0 - nominal["rel0"]) * 1000
                m["injected_max_abs_mm_xyz"] = [round(v, 2) for v in inj.abs().amax(0).tolist()]
                after = (rel1 - nominal["rel"]) * 1000
                m["after_first_step_max_abs_mm_xyz"] = [round(v, 2) for v in after.abs().amax(0).tolist()]
                ph["expected"] = [round(v * 1000, 2) for v in lerp(kcfg.initial, kcfg.final, f)]
            elif knob == "grasp_rot":

                def rpy_deg(q, q_nom):
                    dq = math_utils.quat_mul(math_utils.quat_inv(q_nom.expand_as(q)), q)
                    r, pch, y = math_utils.euler_xyz_from_quat(dq)
                    rpy = torch.stack([math_utils.wrap_to_pi(a) for a in (r, pch, y)], -1)
                    return [round(math.degrees(v), 2) for v in rpy.abs().amax(0).tolist()]

                m["injected_max_abs_deg_rpy"] = rpy_deg(qrel0, nominal["q0"])
                m["after_first_step_max_abs_deg_rpy"] = rpy_deg(qrel1, nominal["q"])
                ph["expected"] = [round(math.degrees(v), 2) for v in lerp(kcfg.initial, kcfg.final, f)]
            elif knob == "socket_pos":
                off = (sock_pos - nominal["sock_pos"]) * 1000
                m["offset_max_abs_mm_xyz"] = [round(v, 2) for v in off.abs().amax(0).tolist()]
                ph["expected"] = [round(v * 1000, 2) for v in lerp(kcfg.initial, kcfg.final, f)]
            elif knob == "socket_rot":
                dq = math_utils.quat_mul(math_utils.quat_inv(nominal["sock_quat"].expand_as(sock_quat)), sock_quat)
                r, pch, y = math_utils.euler_xyz_from_quat(dq)
                rpy = torch.stack([math_utils.wrap_to_pi(a) for a in (r, pch, y)], -1)
                m["offset_max_abs_deg_rpy"] = [round(math.degrees(v), 2) for v in rpy.abs().amax(0).tolist()]
                ph["expected"] = [round(math.degrees(v), 2) for v in lerp(kcfg.initial, kcfg.final, f)]
            elif knob == "action_latency":
                m["latency_steps_seen"] = sorted(set(term._latency_steps.tolist()))
                ph["expected"] = lerp(kcfg.initial, kcfg.final, f)
                per_env_param = term._latency_steps.float()
            elif knob in OBS_TERM or knob == "action_noise":
                ph["expected"] = {
                    "bias_halfwidth": lerp(kcfg.bias_initial, kcfg.bias_final, f),
                    "noise_halfwidth": lerp(kcfg.noise_initial, kcfg.noise_final, f),
                }
            elif knob == "plug_wrench_force":
                ph["expected"] = lerp(kcfg.initial, kcfg.final, f)

            # ---------------- scripted excitation ----------------
            traj, obs_err, act_err, wrench_f, wrench_t = [], [], [], [], []
            ins_dist, rel_start = [], plug_rel()[0]
            for k in range(steps):
                a = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=dev)
                if excite == "insert":
                    s_pos, p_pos = mate_points()
                    if knob in ("obs_socket_pos", "obs_eef_pos"):
                        # Steer on the observed (noisy) position, like a policy: shift the true point by the
                        # observation error of that term (both are in the robot base frame).
                        s_, e_ = offs[OBS_TERM[knob]]
                        tc = cfgs[names.index(OBS_TERM[knob])]
                        delta = math_utils.quat_apply(
                            root_q, om.compute_group("policy")[:, s_:e_] - tc.func(u, **tc.params)
                        )
                        if knob == "obs_socket_pos":
                            s_pos = s_pos + delta
                        else:
                            p_pos = p_pos + delta
                    err_b = math_utils.quat_apply_inverse(root_q, s_pos - p_pos)
                    a[:, 0:3] = torch.clamp(0.5 * err_b / 0.025, -PUSH, PUSH)
                else:
                    a[:, 0] = action_x(k)
                env.step(a)
                if excite == "insert":
                    s_pos, p_pos = mate_points()
                    ins_dist.append(torch.linalg.norm(p_pos - s_pos, dim=-1))
                traj.append(t(robot.data.body_pos_w)[:, eef].clone())
                if knob in OBS_TERM and k < 30:
                    s, e = offs[OBS_TERM[knob]]
                    noisy = om.compute_group("policy")[:, s:e]
                    tc = cfgs[names.index(OBS_TERM[knob])]
                    clean = tc.func(u, **tc.params)
                    obs_err.append(noisy - clean)
                    if markers is not None:
                        markers.visualize(
                            translations=torch.stack([clean[0], noisy[0]]) + origin0, marker_indices=[0, 1]
                        )
                if knob == "action_noise" and k < 30:
                    act_err.append(term.raw_actions - a)
                if knob == "plug_wrench_force":
                    wrench_f.append(t(plug.permanent_wrench_composer.out_force_b).reshape(u.num_envs, -1, 3)[:, 0])
                    wrench_t.append(t(plug.permanent_wrench_composer.out_torque_b).reshape(u.num_envs, -1, 3)[:, 0])

            if excite == "insert":
                d = torch.stack(ins_dist)  # (steps, envs)
                succ = d < u._success_pos_threshold
                first = torch.where(
                    succ.any(0), succ.float().argmax(0), torch.full_like(succ[0], steps, dtype=torch.long)
                )
                rel_end, _ = plug_rel()
                dropped = torch.linalg.norm(rel_end - rel_start, dim=-1) > 0.02
                e_ins = ph.setdefault("effect", {})
                e_ins["insert_success_frac"] = round(succ.any(0).float().mean().item(), 3)
                e_ins["first_success_step_range"] = rng(first.float())
                e_ins["final_mate_dist_mm_range"] = rng(d[-1] * 1000)
                e_ins["seated_steps_mean"] = round(succ.float().sum(0).mean().item(), 1)
                e_ins["dropped"] = int(dropped.sum().item())
                ins_first = first.float()
            traj = torch.stack(traj)  # (steps, envs, 3)
            p0 = traj[4]
            dist = (traj - p0).norm(dim=-1)  # (steps, envs)
            disp_push = dist[14] * 1000
            moved = dist[4:15] > 0.001
            onset = torch.where(moved.any(0), moved.float().argmax(0), torch.full_like(moved[0], 11, dtype=torch.long))
            z_drop = (p0[:, 2] - traj[-1][:, 2]) * 1000
            spread = (traj - p0).std(dim=1).norm(dim=-1).mean().item() * 1000
            e = ph.setdefault("effect", {})
            e["response_spread_mm"] = round(spread, 3)
            e["push_disp_mm_range"] = rng(disp_push)
            e["onset_steps_range"] = rng(onset.float())

            if obs_err:
                err = torch.stack(obs_err)  # (30, envs, d)
                bias = err.mean(0)
                resid = err - bias
                m["bias_max_abs"] = round(bias.abs().max().item(), 5)
                m["noise_max_abs"] = round(resid.abs().max().item(), 5)
            if act_err:
                err = torch.stack(act_err)
                bias = err.mean(0)
                m["bias_max_abs"] = round(bias.abs().max().item(), 5)
                m["noise_max_abs"] = round((err - bias).abs().max().item(), 5)
                m["term_bias_max_abs"] = round(term._action_bias.abs().max().item(), 5)
            if wrench_f:
                wf, wt = torch.stack(wrench_f), torch.stack(wrench_t)
                m["force_component_range_N"] = rng(wf)
                m["torque_component_range_Nm"] = rng(wt)
                m["force_has_both_signs"] = bool((wf > 0).any().item() and (wf < 0).any().item())
            if knob in ("grasp_pos", "grasp_rot", "finger_friction"):
                rel_end, _ = plug_rel()
                settle = (rel_end - nominal["rel"]) * 1000
                e["plug_in_gripper_after_phase_max_abs_mm_xyz"] = [round(v, 2) for v in settle.abs().amax(0).tolist()]

            if level > 0 and excite == "insert":
                if knob in (
                    "osc_stiffness",
                    "osc_damping_ratio",
                    "joint_armature",
                    "joint_friction",
                    "finger_friction",
                    "plug_mass",
                    "action_latency",
                ):
                    e["corr_param_vs_first_success_step"] = corr(per_env_param, ins_first)
            if level > 0 and excite == "push":
                if knob in (
                    "osc_stiffness",
                    "osc_damping_ratio",
                    "joint_armature",
                    "joint_friction",
                    "finger_friction",
                ):
                    e["corr_param_vs_push_disp"] = corr(per_env_param, disp_push)
                if knob == "plug_mass":
                    e["corr_mass_vs_z_drop"] = corr(per_env_param, z_drop)
                if knob == "action_latency":
                    e["corr_latency_vs_onset"] = corr(per_env_param, onset.float())

            result["levels"][str(level)] = ph

        print("SWEEP_RESULT " + json.dumps(result))
        env.close()


if __name__ == "__main__":
    main()
