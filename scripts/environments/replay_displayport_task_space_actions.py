# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay recorded DisplayPort task-space deltas through Newton IK or OSC.

The input CSV must contain six ``action_*`` columns in physical units: position
deltas [m] followed by axis-angle rotation deltas [rad]. These are sent directly
to the action adapter without policy normalization, clipping, or action scaling.

By default, the replay matches the reference PhysX control cadence: a 240 Hz
outer physics step and eight simulation steps per action, yielding exactly
30 Hz task-space commands. Newton uses eight solver substeps per outer step
(1920 Hz) and one collision update per outer step (240 Hz).
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.joint_pos_env_cfg import (
    _SOCKET_ROT,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.displayport_insertion_env_cfg import compute_socket_root
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_IK_TASK = "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Newton-IK"
_OSC_TASK = "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Nominal-DR-Newton-OSC-FlangePose6D"
_ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
_TCP_OFFSET_M = (0.0, 0.0, 0.1925)
_PHYSX_OSC_STIFFNESS = (300.0, 300.0, 300.0, 30.0, 30.0, 30.0)
_PHYSX_OSC_DAMPING_RATIO = (
    35.0 / (2.0 * math.sqrt(300.0)),
    35.0 / (2.0 * math.sqrt(300.0)),
    35.0 / (2.0 * math.sqrt(300.0)),
    1.1 / (2.0 * math.sqrt(30.0)),
    1.1 / (2.0 * math.sqrt(30.0)),
    1.1 / (2.0 * math.sqrt(30.0)),
)
_PHYSX_ARM_GAINS = {
    "shoulder": (1320.0, 72.0),
    "elbow": (600.0, 35.0),
    "wrist": (216.0, 29.0),
}
_PHYSX_2X_ARM_GAINS = {
    name: (2.0 * stiffness, math.sqrt(2.0) * damping) for name, (stiffness, damping) in _PHYSX_ARM_GAINS.items()
}


def _to_torch(value) -> torch.Tensor:
    """Return a Torch view of a Torch, Warp, or proxy array."""
    if hasattr(value, "torch"):
        return value.torch
    if torch.is_tensor(value):
        return value
    return wp.to_torch(value)


def _as_numpy(value) -> np.ndarray:
    """Return environment-zero data as a flat NumPy array."""
    tensor = _to_torch(value)
    if tensor.ndim > 1:
        tensor = tensor[0]
    return tensor.detach().cpu().numpy().reshape(-1)


def _load_trace(path: Path) -> list[dict[str, float]]:
    """Load and validate a physical task-space action trace."""
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {path}")
        required = [*(f"action_{i}" for i in range(6)), *(f"joint_{i}_pos" for i in range(1, 8))]
        missing = [name for name in required if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"Input CSV is missing columns: {missing}")
        rows = [{key: float(value) for key, value in row.items() if value not in (None, "")} for row in reader]
    if not rows:
        raise ValueError(f"Input CSV contains no samples: {path}")
    return rows


def _trace_rate_hz(rows: list[dict[str, float]]) -> float | None:
    """Return the median source sample rate [Hz], when timestamps are present."""
    times = [row.get("sim_time") for row in rows]
    if len(times) < 2 or any(value is None for value in times):
        return None
    delta = np.diff(np.asarray(times, dtype=np.float64))
    valid = delta[delta > 0.0]
    return float(1.0 / np.median(valid)) if valid.size else None


def _socket_geometry_position(row: dict[str, float]) -> tuple[float, float, float]:
    """Read the socket insertion-point position [m] from a trace row."""
    key_sets = (
        ("socket_kp_pos_x", "socket_kp_pos_y", "socket_kp_pos_z"),
        ("obs_9", "obs_10", "obs_11"),
    )
    for keys in key_sets:
        if all(key in row for key in keys):
            return tuple(row[key] for key in keys)
    raise ValueError("Input CSV must contain socket_kp_pos_* or task-space obs_9..obs_11 columns.")


def _place_plug_at_grasp_pose(
    env,
    env_ids: torch.Tensor,
    robot_asset_cfg: SceneEntityCfg,
    target_object_name: str,
    end_effector_body_name: str,
    grasp_offset: list[float],
    grasp_rot_offset: list[float],
    gripper_joint_setter_func,
    hand_hold_width: float,
    hand_close_width: float,
    num_arm_joints: int,
) -> None:
    """Place the plug in the grasp while preserving the configured arm pose."""
    robot = env.scene[robot_asset_cfg.name]
    plug = env.scene[target_object_name]
    body_ids, _ = robot.find_bodies([end_effector_body_name])
    body_id = body_ids[0]
    num_envs = len(env_ids)

    offset = torch.tensor(grasp_offset, device=env.device, dtype=torch.float32).repeat(num_envs, 1)
    rotation_offset = torch.tensor(grasp_rot_offset, device=env.device, dtype=torch.float32).repeat(num_envs, 1)
    body_pos = _to_torch(robot.data.body_pos_w)[env_ids, body_id].clone()
    body_quat = _to_torch(robot.data.body_quat_w)[env_ids, body_id].clone()
    plug_quat = math_utils.quat_mul(body_quat, math_utils.quat_conjugate(rotation_offset))
    plug_pos = body_pos - math_utils.quat_apply(body_quat, offset)
    plug.write_root_pose_to_sim_index(root_pose=torch.cat((plug_pos, plug_quat), dim=-1), env_ids=env_ids)
    plug.write_root_velocity_to_sim_index(root_velocity=torch.zeros((num_envs, 6), device=env.device), env_ids=env_ids)

    all_joint_ids, all_joint_names = robot.find_joints([".*"])
    finger_joint_ids = all_joint_ids[num_arm_joints:]
    joint_name_to_idx = dict(zip(all_joint_names, all_joint_ids))
    joint_pos = _to_torch(robot.data.joint_pos)[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    setter_args = (joint_pos, list(range(num_envs)), finger_joint_ids)
    if "joint_name_to_idx" in inspect.signature(gripper_joint_setter_func).parameters:
        gripper_joint_setter_func(*setter_args, hand_hold_width, joint_name_to_idx)
    else:
        gripper_joint_setter_func(*setter_args, hand_hold_width)
    robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

    if "joint_name_to_idx" in inspect.signature(gripper_joint_setter_func).parameters:
        gripper_joint_setter_func(*setter_args, hand_close_width, joint_name_to_idx)
    else:
        gripper_joint_setter_func(*setter_args, hand_close_width)
    robot.set_joint_position_target_index(target=joint_pos, joint_ids=all_joint_ids, env_ids=env_ids)


def _disable_rollout_mutations(cfg) -> None:
    """Disable randomization and automatic termination for open-loop replay."""
    for name in (
        "randomize_socket_pose",
        "reset_plug_curriculum",
        "randomize_plug_pose",
        "randomize_arm_joint_friction",
        "randomize_arm_pd_gains",
    ):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)
    for name in ("time_out", "plug_dropped", "plug_orientation_exceeded", "plug_overtravel"):
        if hasattr(cfg.terminations, name):
            setattr(cfg.terminations, name, None)
    cfg.physics_watchdog_enabled = False
    if hasattr(cfg.observations, "policy") and hasattr(cfg.observations.policy, "enable_corruption"):
        cfg.observations.policy.enable_corruption = False


def _configure_initial_state(cfg, first_row: dict[str, float]) -> tuple[float, float, float]:
    """Set the recorded arm joints and socket pose, then retain a physical grasp."""
    joint_pos = dict(cfg.scene.robot.init_state.joint_pos)
    joint_pos.update({name: first_row[f"joint_{index}_pos"] for index, name in enumerate(_ARM_JOINT_NAMES, 1)})
    cfg.scene.robot.init_state.joint_pos = joint_pos

    socket_geometry_pos = _socket_geometry_position(first_row)
    cfg.scene.dp_socket.init_state.pos = compute_socket_root(socket_geometry_pos, _SOCKET_ROT)
    cfg.scene.dp_socket.init_state.rot = _SOCKET_ROT
    if hasattr(cfg, "fixed_asset_init_pos_center"):
        cfg.fixed_asset_init_pos_center = list(socket_geometry_pos)

    grasp_event = getattr(cfg.events, "set_robot_to_grasp_pose", None)
    if grasp_event is None:
        raise ValueError("Task does not define set_robot_to_grasp_pose; cannot preserve the recorded grasp.")
    params = grasp_event.params
    cfg.events.set_robot_to_grasp_pose = None
    cfg.events.place_plug_in_grasp = EventTerm(
        func=_place_plug_at_grasp_pose,
        mode="reset",
        params={
            "robot_asset_cfg": params.get("robot_asset_cfg", SceneEntityCfg("robot")),
            "target_object_name": params["target_object_name"],
            "end_effector_body_name": params["end_effector_body_name"],
            "grasp_offset": params.get("grasp_offset", [0.0, 0.0, 0.0]),
            "grasp_rot_offset": params["grasp_rot_offset"],
            "gripper_joint_setter_func": params["gripper_joint_setter_func"],
            "hand_hold_width": cfg.hand_hold_width,
            "hand_close_width": cfg.hand_close_width,
            "num_arm_joints": params["num_arm_joints"],
        },
    )
    return socket_geometry_pos


def _configure_physical_action_input(cfg, controller: str, osc_profile: str) -> None:
    """Make the action adapter accept physical deltas without scaling or clipping."""
    action_cfg = cfg.actions.arm_action
    action_cfg.clip = None
    if controller == "ik":
        pose_objectives = [objective for objective in action_cfg.objectives if hasattr(objective, "body_name")]
        if len(pose_objectives) != 1 or pose_objectives[0].body_name != "flange":
            raise ValueError("IK replay requires one flange pose objective.")
        pose_objectives[0].scale = 1.0
        return

    action_cfg.position_scale = 1.0
    action_cfg.orientation_scale = 1.0
    if osc_profile == "physx":
        controller_cfg = action_cfg.controller_cfg
        controller_cfg.inertial_dynamics_decoupling = False
        controller_cfg.motion_stiffness_task = _PHYSX_OSC_STIFFNESS
        controller_cfg.motion_damping_ratio_task = _PHYSX_OSC_DAMPING_RATIO


def _configure_arm_gain_profile(cfg, controller: str, profile: str) -> dict[str, tuple[float, float]]:
    """Select implicit arm gains for an IK replay and return the resolved values."""
    if controller == "osc":
        if profile != "current":
            raise ValueError("OSC replay uses direct effort control and requires arm_gain_profile=current.")
    elif profile != "current":
        gain_map = _PHYSX_ARM_GAINS if profile == "physx" else _PHYSX_2X_ARM_GAINS
        for actuator_name, (stiffness, damping) in gain_map.items():
            actuator_cfg = cfg.scene.robot.actuators[actuator_name]
            actuator_cfg.stiffness = stiffness
            actuator_cfg.damping = damping

    return {
        name: (
            float(cfg.scene.robot.actuators[name].stiffness),
            float(cfg.scene.robot.actuators[name].damping),
        )
        for name in _PHYSX_ARM_GAINS
    }


def _body_pose(robot, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return an articulation body pose for environment zero."""
    body_ids, _ = robot.find_bodies([body_name])
    body_id = body_ids[0]
    pos = _as_numpy(robot.data.body_link_pos_w[:, body_id])
    quat = _as_numpy(robot.data.body_link_quat_w[:, body_id])
    return pos, quat


def _tcp_pose(flange_pos: np.ndarray, flange_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the 192.5 mm TCP pose for one flange pose."""
    quat = torch.tensor(flange_quat, dtype=torch.float32).unsqueeze(0)
    offset = torch.tensor(_TCP_OFFSET_M, dtype=torch.float32).unsqueeze(0)
    rotated = math_utils.quat_apply(quat, offset)[0].numpy()
    return flange_pos + rotated, flange_quat


def _snapshot(env) -> dict[str, np.ndarray]:
    """Capture controller, articulation, and object telemetry for environment zero."""
    base = env.unwrapped
    robot = base.scene["robot"]
    plug = base.scene["dp_plug"]
    flange_pos, flange_quat = _body_pose(robot, "flange")
    tcp_pos, tcp_quat = _tcp_pose(flange_pos, flange_quat)
    result = {
        "joint_pos": _as_numpy(robot.data.joint_pos)[:7],
        "joint_vel": _as_numpy(robot.data.joint_vel)[:7],
        "joint_pos_target": _as_numpy(robot.data.joint_pos_target)[:7],
        "computed_torque": _as_numpy(robot.data.computed_torque)[:7],
        "applied_torque": _as_numpy(robot.data.applied_torque)[:7],
        "flange_pos": flange_pos,
        "flange_quat": flange_quat,
        "tcp_pos": tcp_pos,
        "tcp_quat": tcp_quat,
        "plug_pos": _as_numpy(plug.data.root_pos_w)[:3],
        "plug_quat": _as_numpy(plug.data.root_quat_w)[:4],
        "plug_lin_vel": _as_numpy(plug.data.root_lin_vel_w)[:3],
        "plug_ang_vel": _as_numpy(plug.data.root_ang_vel_w)[:3],
    }
    action_term = base.action_manager.get_term("arm_action")
    result["processed_action"] = _as_numpy(action_term.processed_actions)[:6]
    joint_pos_des = getattr(action_term, "_joint_pos_des", None)
    if joint_pos_des is not None:
        result["controller_joint_target"] = _as_numpy(joint_pos_des)[:7]
    joint_efforts = getattr(action_term, "_joint_efforts", None)
    if joint_efforts is not None:
        result["controller_joint_effort"] = _as_numpy(joint_efforts)[:7]
    return result


def _add_vector(row: dict[str, Any], prefix: str, values: np.ndarray | list[float]) -> None:
    """Add one flattened vector to a telemetry row."""
    for index, value in enumerate(values):
        row[f"{prefix}_{index}"] = float(value)


def _source_state(row: dict[str, float]) -> dict[str, np.ndarray]:
    """Extract comparable state vectors from one PhysX trace row."""
    result = {
        "joint_pos": np.asarray([row[f"joint_{i}_pos"] for i in range(1, 8)], dtype=np.float64),
        "action": np.asarray([row[f"action_{i}"] for i in range(6)], dtype=np.float64),
    }
    for prefix, keys in {
        "flange_pos": ("flange_pos_x", "flange_pos_y", "flange_pos_z"),
        "plug_pos": ("plug_pos_x", "plug_pos_y", "plug_pos_z"),
    }.items():
        if all(key in row for key in keys):
            result[prefix] = np.asarray([row[key] for key in keys], dtype=np.float64)
    return result


def _write_summary(
    output_dir: Path,
    rows: list[dict[str, Any]],
    source_rate_hz: float | None,
    config: dict[str, Any],
) -> None:
    """Write aggregate tracking and saturation diagnostics."""
    joint_error = np.asarray(
        [[row[f"pre_joint_pos_{i}"] - row[f"source_joint_pos_{i}"] for i in range(7)] for row in rows]
    )
    flange_error = np.asarray(
        [[row[f"pre_flange_pos_{i}"] - row[f"source_flange_pos_{i}"] for i in range(3)] for row in rows]
    )
    plug_error = np.asarray(
        [[row[f"pre_plug_pos_{i}"] - row[f"source_plug_pos_{i}"] for i in range(3)] for row in rows]
    )
    action_error = np.asarray(
        [[row[f"post_processed_action_{i}"] - row[f"source_action_{i}"] for i in range(6)] for row in rows]
    )
    applied_torque = np.asarray([[row[f"post_applied_torque_{i}"] for i in range(7)] for row in rows])
    joint_vel = np.asarray([[row[f"post_joint_vel_{i}"] for i in range(7)] for row in rows])
    summary = {
        **config,
        "num_samples": len(rows),
        "source_action_rate_hz": source_rate_hz,
        "joint_position_rmse_rad": float(np.sqrt(np.mean(np.square(joint_error)))),
        "joint_position_max_abs_error_rad": float(np.max(np.abs(joint_error))),
        "flange_position_rmse_m": float(np.sqrt(np.mean(np.square(flange_error)))),
        "flange_position_max_norm_error_m": float(np.max(np.linalg.vector_norm(flange_error, axis=1))),
        "plug_position_rmse_m": float(np.sqrt(np.mean(np.square(plug_error)))),
        "plug_position_max_norm_error_m": float(np.max(np.linalg.vector_norm(plug_error, axis=1))),
        "processed_action_max_abs_error": float(np.max(np.abs(action_error))),
        "max_applied_torque_abs": [float(value) for value in np.max(np.abs(applied_torque), axis=0)],
        "max_joint_velocity_abs_rad_s": [float(value) for value in np.max(np.abs(joint_vel), axis=0)],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print("\n[SUMMARY]")
    print(json.dumps(summary, indent=2))


def main() -> None:
    """Run one deterministic open-loop task-space replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", type=Path, required=True, help="PhysX task-space telemetry CSV.")
    parser.add_argument("--controller", choices=("ik", "osc"), required=True, help="Newton controller adapter.")
    parser.add_argument("--task", default=None, help="Task override; defaults to a nominal flange-action task.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory for telemetry and summary files.")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum trace samples to replay.")
    parser.add_argument("--physics_preset", default="newton_sdf", help="Newton physics preset.")
    parser.add_argument("--physics_dt", type=float, default=1.0 / 240.0, help="Outer physics step [s].")
    parser.add_argument("--decimation", type=int, default=8, help="Outer physics steps per task-space command.")
    parser.add_argument("--num_substeps", type=int, default=8, help="Newton solver substeps per outer step.")
    parser.add_argument(
        "--collision_decimation",
        type=int,
        default=0,
        help="Solver substeps between mid-step collision updates; zero means once per outer step.",
    )
    parser.add_argument("--update_data_interval", type=int, default=8, help="MuJoCo-Warp data update interval.")
    parser.add_argument(
        "--osc_profile",
        choices=("current", "physx"),
        default="current",
        help="Use current Newton OSC settings or match the source PhysX OSC gains/decoupling.",
    )
    parser.add_argument(
        "--arm_gain_profile",
        choices=("current", "physx", "physx_2x"),
        default="current",
        help="Implicit arm-gain profile for Newton IK; OSC always uses zero arm PD gains.",
    )
    parser.add_argument(
        "--match_source_plug_position",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After grasp placement, overwrite plug position with CSV row zero while retaining grasp orientation.",
    )
    parser.add_argument("--print_interval", type=int, default=20, help="Samples between progress messages.")
    add_launcher_args(parser)
    args = parser.parse_args()

    input_path = args.input_csv.expanduser().resolve()
    trace = _load_trace(input_path)
    source_rate_hz = _trace_rate_hz(trace)
    if args.max_steps is not None:
        trace = trace[: args.max_steps]
    task = args.task or (_IK_TASK if args.controller == "ik" else _OSC_TASK)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("artifacts/displayport_controller_replay") / input_path.stem / (
        f"newton_{args.controller}_{args.osc_profile}_{timestamp}"
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg = resolve_presets(cfg, {args.physics_preset})
    cfg.scene.num_envs = 1
    cfg.seed = 42
    cfg.sim.device = args.device or "cuda:0"
    cfg.sim.dt = args.physics_dt
    cfg.decimation = args.decimation
    cfg.sim.render_interval = args.decimation
    cfg.sim.physics.num_substeps = args.num_substeps
    cfg.sim.physics.collision_decimation = args.collision_decimation
    solver_cfg = cfg.sim.physics.solver_cfg
    if hasattr(solver_cfg, "update_data_interval"):
        solver_cfg.update_data_interval = args.update_data_interval
    else:
        for entry in getattr(solver_cfg, "entries", ()):
            if hasattr(entry.solver_cfg, "update_data_interval"):
                entry.solver_cfg.update_data_interval = args.update_data_interval
    _disable_rollout_mutations(cfg)
    socket_geometry_pos = _configure_initial_state(cfg, trace[0])
    _configure_physical_action_input(cfg, args.controller, args.osc_profile)
    arm_gains = _configure_arm_gain_profile(cfg, args.controller, args.arm_gain_profile)

    collision_passes = 1
    if 0 < args.collision_decimation < args.num_substeps:
        collision_passes += (args.num_substeps - 1) // args.collision_decimation
    config = {
        "input_csv": str(input_path),
        "task": task,
        "controller": args.controller,
        "osc_profile": args.osc_profile if args.controller == "osc" else None,
        "arm_gain_profile": args.arm_gain_profile,
        "arm_gains": arm_gains,
        "physics_preset": args.physics_preset,
        "physics_dt_s": args.physics_dt,
        "outer_physics_rate_hz": 1.0 / args.physics_dt,
        "action_rate_hz": 1.0 / (args.physics_dt * args.decimation),
        "solver_rate_hz": args.num_substeps / args.physics_dt,
        "collision_rate_hz": collision_passes / args.physics_dt,
        "num_substeps": args.num_substeps,
        "collision_decimation": args.collision_decimation,
        "update_data_interval": args.update_data_interval,
        "socket_geometry_pos_m": socket_geometry_pos,
        "tcp_offset_m": _TCP_OFFSET_M,
        "physical_action_input": True,
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)

    print("\n[DISPLAYPORT TASK-SPACE REPLAY]")
    print(f"input:          {input_path}")
    print(f"controller:     {args.controller}")
    print(f"task:           {task}")
    print(f"source rate:    {source_rate_hz:.3f} Hz" if source_rate_hz else "source rate:    unavailable")
    print(f"action rate:    {config['action_rate_hz']:.3f} Hz")
    print(f"outer physics:  {config['outer_physics_rate_hz']:.1f} Hz")
    print(f"solver:         {config['solver_rate_hz']:.1f} Hz")
    print(f"collision:      {config['collision_rate_hz']:.1f} Hz")
    print(f"output:         {output_dir}")

    telemetry: list[dict[str, Any]] = []
    with launch_simulation(cfg, args):
        env = gym.make(task, cfg=cfg)
        env.reset()
        base = env.unwrapped
        plug = base.scene["dp_plug"]

        if args.match_source_plug_position and all(
            key in trace[0] for key in ("plug_pos_x", "plug_pos_y", "plug_pos_z")
        ):
            plug_pos = torch.tensor(
                [trace[0]["plug_pos_x"], trace[0]["plug_pos_y"], trace[0]["plug_pos_z"]],
                device=base.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            plug_pos += base.scene.env_origins[0:1]
            plug_quat = _to_torch(plug.data.root_quat_w)[0:1].clone()
            plug.write_root_pose_to_sim_index(root_pose=torch.cat((plug_pos, plug_quat), dim=-1))
            plug.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=base.device))

        with torch.inference_mode():
            for step, source_row in enumerate(trace):
                source = _source_state(source_row)
                pre = _snapshot(env)
                action = torch.tensor(source["action"], device=base.device, dtype=torch.float32).unsqueeze(0)
                env.step(action)
                post = _snapshot(env)

                row: dict[str, Any] = {
                    "step": step,
                    "source_time_s": source_row.get("sim_time", step / 30.0),
                    "replay_time_s": step * args.physics_dt * args.decimation,
                }
                for name, values in source.items():
                    _add_vector(row, f"source_{name}", values)
                for phase, snapshot in (("pre", pre), ("post", post)):
                    for name, values in snapshot.items():
                        _add_vector(row, f"{phase}_{name}", values)
                telemetry.append(row)

                if step == 0 or (step + 1) % args.print_interval == 0 or step + 1 == len(trace):
                    flange_error_mm = 1000.0 * np.linalg.vector_norm(pre["flange_pos"] - source["flange_pos"])
                    joint_error = np.linalg.vector_norm(pre["joint_pos"] - source["joint_pos"])
                    print(
                        f"step={step:3d} t={row['replay_time_s']:.3f}s "
                        f"flange_err={flange_error_mm:8.3f} mm joint_err={joint_error:8.4f} rad"
                    )
        env.close()

    fieldnames = list(telemetry[0])
    with (output_dir / "telemetry.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(telemetry)
    _write_summary(output_dir, telemetry, source_rate_hz, config)
    print(f"[INFO] Wrote {len(telemetry)} samples to {output_dir / 'telemetry.csv'}")


if __name__ == "__main__":
    main()
