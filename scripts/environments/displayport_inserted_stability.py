# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure DisplayPort plug drift from the authored seated pose under Newton physics."""

import argparse
import os

import gymnasium as gym
import torch
import warp as wp

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.joint_pos_env_cfg import (
    _GEOMETRY_POS,
    _SOCKET_ROT,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.displayport_insertion_env_cfg import (
    DISPLAY_ASSETS_DIR,
    compute_plug_pose,
    compute_socket_root,
)
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

TASK_NAME = "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav"

_POINT_SOCKET_ASSETS = {
    "fixed": "display_port_socket_newton_sdf.usda",
    "legacy": "display_port_socket_newton_sdf_legacy.usda",
}
_HYDRO_SOCKET_ASSETS = {
    "fixed": "display_port_socket_newton_hydroelastic.usda",
    "legacy": "display_port_socket_newton_hydroelastic_legacy.usda",
}
_POINT_GAP_ASSETS = {
    "5mm": ("display_port_plug_newton_sdf.usda", "display_port_socket_newton_sdf.usda", 0.005),
    "1mm": (
        "display_port_plug_newton_sdf_gap_1mm.usda",
        "display_port_socket_newton_sdf_gap_1mm.usda",
        0.001,
    ),
    "0.5mm": (
        "display_port_plug_newton_sdf_gap_0p5mm.usda",
        "display_port_socket_newton_sdf_gap_0p5mm.usda",
        0.0005,
    ),
}


def _quaternion_error_deg(quat: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Return the shortest angular distance between unit quaternions [deg]."""
    dot = torch.sum(quat * reference, dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot) * (180.0 / torch.pi)


def _select_assets(preset_name: str, socket_variant: str, sdf_gap: str) -> tuple[str, str, float]:
    """Return plug path, socket path, and authored per-shape gap [m]."""
    if preset_name == "newton_hydroelastic":
        if sdf_gap != "5mm":
            raise ValueError("Hydroelastic comparison currently supports only the canonical 5 mm gap.")
        plug_filename = "display_port_plug_newton_hydroelastic.usda"
        socket_filename = _HYDRO_SOCKET_ASSETS[socket_variant]
        gap = 0.005
    elif socket_variant == "legacy":
        if sdf_gap != "5mm":
            raise ValueError("Legacy-socket comparison currently supports only the canonical 5 mm gap.")
        plug_filename = "display_port_plug_newton_sdf.usda"
        socket_filename = _POINT_SOCKET_ASSETS[socket_variant]
        gap = 0.005
    else:
        plug_filename, socket_filename, gap = _POINT_GAP_ASSETS[sdf_gap]

    return (
        os.path.join(DISPLAY_ASSETS_DIR, plug_filename),
        os.path.join(DISPLAY_ASSETS_DIR, socket_filename),
        gap,
    )


def main() -> None:
    """Load a seated asset pair and report drift from the authored mate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200, help="Number of policy steps to simulate.")
    parser.add_argument("--report_interval", type=int, default=25, help="Policy steps between progress reports.")
    parser.add_argument(
        "--physics_preset",
        choices=("newton_sdf", "newton_hydroelastic"),
        default="newton_sdf",
        help="Newton SDF contact pipeline to test.",
    )
    parser.add_argument(
        "--socket_variant",
        choices=("fixed", "legacy"),
        default="fixed",
        help="Use the cleaned socket or the previous split-visuals socket.",
    )
    parser.add_argument(
        "--sdf_gap",
        choices=tuple(_POINT_GAP_ASSETS),
        default="5mm",
        help="Authored contact gap on each mating SDF shape.",
    )
    parser.add_argument(
        "--clearance",
        type=float,
        default=0.0,
        help="Distance [m] to retract the plug from the exact CAD mate along the insertion axis.",
    )
    parser.add_argument(
        "--dynamic_plug",
        action="store_true",
        help="Let the ungrasped plug move instead of freezing it at the seated pose.",
    )
    parser.add_argument(
        "--disable_gravity",
        action="store_true",
        help="Disable scene gravity to isolate the initial plug-socket contact response.",
    )
    parser.add_argument(
        "--isolate_robot",
        action="store_true",
        help="Move the robot 2 m away so it cannot affect the free-dynamics test.",
    )
    parser.add_argument("--sim_dt", type=float, default=0.01, help="Outer simulation step [s].")
    parser.add_argument("--num_substeps", type=int, default=20, help="Solver substeps per outer simulation step.")
    parser.add_argument(
        "--collision_decimation",
        type=int,
        default=10,
        help="Solver substeps between Newton collision-pipeline updates.",
    )
    parser.add_argument(
        "--update_data_interval",
        type=int,
        default=10,
        help="Solver substeps between MuJoCo-Warp data updates.",
    )
    parser.add_argument("--policy_decimation", type=int, default=3, help="Outer steps per zero action.")
    parser.add_argument(
        "--fail_on_unstable",
        action="store_true",
        help="Exit with an error when dynamic drift exceeds 1 mm or 0.25 degrees.",
    )
    add_launcher_args(parser)
    args = parser.parse_args()

    plug_asset, socket_asset, shape_gap = _select_assets(args.physics_preset, args.socket_variant, args.sdf_gap)
    cfg = load_cfg_from_registry(TASK_NAME, "env_cfg_entry_point")
    cfg = resolve_presets(cfg, {args.physics_preset})
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device or "cuda:0"
    cfg.seed = 42

    cfg.sim.dt = args.sim_dt
    cfg.sim.physics.num_substeps = args.num_substeps
    cfg.sim.physics.collision_decimation = args.collision_decimation
    cfg.sim.physics.solver_cfg.update_data_interval = args.update_data_interval
    cfg.decimation = args.policy_decimation
    cfg.sim.render_interval = args.policy_decimation
    cfg.scene.dp_plug.spawn.usd_path = plug_asset
    cfg.scene.dp_socket.spawn.usd_path = socket_asset

    cfg.events.randomize_socket_pose = None
    cfg.events.reset_plug_curriculum = None
    cfg.events.set_robot_to_grasp_pose = None
    cfg.events.randomize_arm_joint_friction = None
    cfg.events.randomize_arm_pd_gains = None
    cfg.terminations.time_out = None
    cfg.terminations.plug_dropped = None
    cfg.terminations.plug_orientation_exceeded = None
    cfg.observations.policy.enable_corruption = False
    cfg.scene.dp_plug.spawn.rigid_props.kinematic_enabled = not args.dynamic_plug
    if args.disable_gravity:
        cfg.sim.gravity = (0.0, 0.0, 0.0)
    if args.isolate_robot:
        cfg.scene.robot.init_state.pos = (-2.0, 0.0, 0.0)

    socket_root = compute_socket_root(_GEOMETRY_POS, _SOCKET_ROT)
    plug_root, plug_rot = compute_plug_pose(_GEOMETRY_POS, _SOCKET_ROT, z_clearance=args.clearance)
    cfg.scene.dp_socket.init_state.pos = socket_root
    cfg.scene.dp_socket.init_state.rot = _SOCKET_ROT
    cfg.scene.dp_plug.init_state.pos = plug_root
    cfg.scene.dp_plug.init_state.rot = plug_rot

    with launch_simulation(cfg, args):
        env = gym.make(TASK_NAME, cfg=cfg)
        env.reset()
        unwrapped = env.unwrapped
        plug = unwrapped.scene["dp_plug"]
        actions = torch.zeros(env.action_space.shape, device=unwrapped.device)

        target_pos = torch.tensor(plug_root, device=unwrapped.device, dtype=torch.float32).unsqueeze(0)
        target_pos += unwrapped.scene.env_origins
        target_quat = torch.tensor(plug_rot, device=unwrapped.device, dtype=torch.float32).unsqueeze(0)
        max_pos_error = torch.zeros(1, device=unwrapped.device)
        max_angle_error = torch.zeros(1, device=unwrapped.device)
        finite = True

        collision_passes = 1
        if 0 < args.collision_decimation < args.num_substeps:
            collision_passes += (args.num_substeps - 1) // args.collision_decimation

        print("\n[DISPLAYPORT INSERTED-POSE CHECK]")
        print(f"mode: {'dynamic, ungrasped' if args.dynamic_plug else 'kinematic CAD mate'}")
        print(f"scene gravity: {'disabled' if args.disable_gravity else 'enabled'}")
        print(f"robot layout: {'isolated at x=-2 m' if args.isolate_robot else 'training pose'}")
        print(f"contact pipeline: {args.physics_preset}")
        print(f"socket variant: {args.socket_variant}")
        print(f"asset plug:   {plug_asset}")
        print(f"asset socket: {socket_asset}")
        print(f"authored SDF gap: {shape_gap * 1000.0:.3f} mm per shape, {2.0 * shape_gap * 1000.0:.3f} mm per pair")
        print(f"outer rate: {1.0 / args.sim_dt:.1f} Hz")
        print(f"solver rate: {args.num_substeps / args.sim_dt:.1f} Hz")
        print(f"collision rate: {collision_passes / args.sim_dt:.1f} Hz")
        print(f"policy rate: {1.0 / (args.sim_dt * args.policy_decimation):.1f} Hz")
        print(f"insertion clearance: {args.clearance * 1000.0:.3f} mm")

        with torch.inference_mode():
            for step in range(1, args.steps + 1):
                env.step(actions)
                pos = wp.to_torch(plug.data.root_pos_w)
                quat = wp.to_torch(plug.data.root_quat_w)
                state = torch.cat((pos, quat), dim=-1)
                finite = finite and bool(torch.isfinite(state).all())
                pos_error = torch.linalg.vector_norm(pos - target_pos, dim=-1)
                angle_error = _quaternion_error_deg(quat, target_quat)
                max_pos_error = torch.maximum(max_pos_error, pos_error)
                max_angle_error = torch.maximum(max_angle_error, angle_error)

                if step == 1 or step % args.report_interval == 0 or step == args.steps:
                    print(
                        f"step={step:4d} t={step * args.sim_dt * args.policy_decimation:6.3f}s "
                        f"pos_err={pos_error.item() * 1000.0:9.4f} mm "
                        f"angle_err={angle_error.item():9.4f} deg"
                    )
                if not finite:
                    break

        max_pos_error_mm = max_pos_error.item() * 1000.0
        max_angle_error_deg = max_angle_error.item()
        stable = finite and max_pos_error_mm < 1.0 and max_angle_error_deg < 0.25
        print("\n[RESULT]")
        print(f"finite state:          {finite}")
        print(f"max position drift:    {max_pos_error_mm:.4f} mm")
        print(f"max orientation drift: {max_angle_error_deg:.4f} deg")
        print(f"stable (<1 mm, <0.25 deg): {stable}")
        if args.dynamic_plug:
            print("note: dynamic mode removes gripper support and isolates the free-fit contact equilibrium")
        env.close()

        if args.fail_on_unstable and args.dynamic_plug and not stable:
            raise RuntimeError("DisplayPort seated-pose stability limits were exceeded.")


if __name__ == "__main__":
    main()
