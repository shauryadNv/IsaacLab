# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint step-response test for a simulated robot arm.

Records the actual joint trajectory after applying a position step command to
each arm joint individually.  The robot configuration (actuator model, PD
gains, physics properties, dt) is pulled directly from the training
environment config so the test always matches training.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tools/step_response_test_sim.py --headless

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# -- CLI arguments -----------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Joint step-response test – records per-joint transients in simulation."
)
parser.add_argument(
    "--output-dir",
    default="./step_response_results_sim",
    help="Root directory for result plots (default: %(default)s).",
)
parser.add_argument(
    "--sim-dt",
    type=float,
    default=None,
    help="Override the physics timestep [s]. When omitted, uses the value from the training config.",
)
parser.add_argument(
    "--record-duration",
    type=float,
    default=2.0,
    help="Recording window after each step command [s] (default: %(default)s).",
)
parser.add_argument(
    "--settle-duration",
    type=float,
    default=1.5,
    help="Settling time between step tests [s] (default: %(default)s).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os
import re
from collections import defaultdict

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

##
# Training-environment config -- single source of truth
##
# isort: off
from isaaclab_tasks.manager_based.manipulation.deploy.gear_assembly.config.rizon_4s.joint_pos_env_cfg import (
    Rizon4sGearAssemblyEnvCfg,
)

# isort: on

ENV_CFG = Rizon4sGearAssemblyEnvCfg()
ROBOT_CFG = ENV_CFG.scene.robot
TRAINING_DT = ENV_CFG.sim.dt
TRAINING_DECIMATION = ENV_CFG.decimation

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("matplotlib is required – install with: pip install matplotlib")


##
# Constants
##

ARM_JOINT_NAMES: list[str] = [f"joint{i}" for i in range(1, 8)]
NUM_ARM_JOINTS: int = len(ARM_JOINT_NAMES)
STEP_SIZES_DEG: list[float] = [-1.0, 1.0, -5.0, 5.0]
BASELINE_DURATION: float = 0.1  # seconds of baseline recording before the step (matches real robot)


def _training_init_pose_rad() -> list[float]:
    """Return the arm-joint initial pose [rad] defined in the training config."""
    joint_pos_map: dict = ROBOT_CFG.init_state.joint_pos
    return [float(joint_pos_map.get(name, 0.0)) for name in ARM_JOINT_NAMES]


STARTING_POSES: dict = {
    "zeros": {
        "label": "All Zeros",
        "joint_pos_rad": [0.0] * NUM_ARM_JOINTS,
    },
    "training_home": {
        "label": "Training Home",
        "joint_pos_rad": _training_init_pose_rad(),
    },
}


##
# Scene setup
##


def design_scene() -> dict[str, Articulation]:
    """Create a minimal scene using the training-environment robot config.

    The :class:`ArticulationCfg` is taken from
    :class:`Rizon4sGearAssemblyEnvCfg` so that actuators, physics properties,
    and solver settings are identical to the training environment.
    """
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    robot_cfg = ROBOT_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(cfg=robot_cfg)
    return {"robot": robot}


##
# Config introspection helpers
##


def _extract_actuator_info() -> dict[str, dict]:
    """Read actuator type, stiffness, damping for each arm joint from the robot config.

    Returns:
        Mapping from joint name to a dict with keys ``type``, ``stiffness``,
        ``damping``, and (when applicable) ``saturation_effort``.
    """
    info: dict[str, dict] = {}
    for _group, act_cfg in ROBOT_CFG.actuators.items():
        for jname in ARM_JOINT_NAMES:
            if any(re.fullmatch(expr, jname) for expr in act_cfg.joint_names_expr):
                entry: dict = {
                    "type": type(act_cfg).__name__,
                    "stiffness": act_cfg.stiffness,
                    "damping": act_cfg.damping,
                }
                if hasattr(act_cfg, "saturation_effort"):
                    entry["saturation_effort"] = act_cfg.saturation_effort
                info[jname] = entry
    return info


def _print_config_summary(sim_dt: float) -> dict[str, dict]:
    """Print a human-readable summary of the configuration and return gain info."""
    act_info = _extract_actuator_info()

    print("\n" + "=" * 64)
    print(" Step-Response Config (from Rizon4sGearAssemblyEnvCfg)")
    print("=" * 64)
    print(f"  Physics dt        : {sim_dt * 1e3:.3f} ms  (training: {TRAINING_DT * 1e3:.3f} ms)")
    print(f"  Training decimation: {TRAINING_DECIMATION}")
    print(f"  disable_gravity   : {ROBOT_CFG.spawn.rigid_props.disable_gravity}")
    print(f"  solver_pos_iters  : {ROBOT_CFG.spawn.articulation_props.solver_position_iteration_count}")

    first = next(iter(act_info.values()))
    print(f"  Actuator model    : {first['type']}")

    for jname in ARM_JOINT_NAMES:
        a = act_info[jname]
        sat = f"  sat={a['saturation_effort']}" if "saturation_effort" in a else ""
        print(f"    {jname}: Kp={a['stiffness']:<8.1f} Kd={a['damping']:<8.1f}{sat}")

    print("  Starting poses:")
    for key, pose in STARTING_POSES.items():
        degs = [f"{math.degrees(r):.1f}" for r in pose["joint_pos_rad"]]
        print(f"    {key}: {degs}")
    print("=" * 64 + "\n")

    return act_info


##
# Robot helpers
##


def _resolve_arm_joint_ids(robot: Articulation) -> list[int]:
    """Return the simulation-side joint indices for the 7 arm joints."""
    ids = []
    for jname in ARM_JOINT_NAMES:
        found, _ = robot.find_joints(jname)
        ids.append(found[0])
    return ids


def _build_home_pos(robot: Articulation, arm_joint_ids: list[int], arm_rad: list[float]) -> torch.Tensor:
    """Build a full joint-position tensor with the desired arm pose.

    Gripper joints keep their USD defaults; only arm joints are overridden.
    """
    home = wp.to_torch(robot.data.default_joint_pos).clone()
    for idx, rad_val in zip(arm_joint_ids, arm_rad):
        home[:, idx] = rad_val
    return home


def _reset_robot(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    home_pos: torch.Tensor,
    sim_dt: float,
    settle_steps: int,
) -> None:
    """Teleport the robot to *home_pos* and let the controller settle."""
    zero_vel = torch.zeros_like(wp.to_torch(robot.data.default_joint_vel))
    robot.write_joint_position_to_sim_index(position=home_pos)
    robot.write_joint_velocity_to_sim_index(velocity=zero_vel)
    robot.reset()

    for _ in range(settle_steps):
        robot.set_joint_position_target_index(target=home_pos)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim_dt)


##
# Core test loop
##


def run_step_response(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    output_dir: str,
    sim_dt: float,
    record_dur: float,
    settle_dur: float,
) -> None:
    """Run step-response tests on each arm joint for every starting pose."""
    arm_joint_ids = _resolve_arm_joint_ids(robot)
    act_info = _print_config_summary(sim_dt)

    record_steps = int(record_dur / sim_dt)
    settle_steps = int(settle_dur / sim_dt)

    for pose_key, pose_info in STARTING_POSES.items():
        pose_label = pose_info["label"]
        arm_rad = pose_info["joint_pos_rad"]
        pose_dir = os.path.join(output_dir, pose_key)
        os.makedirs(pose_dir, exist_ok=True)

        home_pos = _build_home_pos(robot, arm_joint_ids, arm_rad)

        print(f"\n{'#' * 64}")
        print(f"# Starting pose: {pose_label}")
        print(f"#   {dict(zip(ARM_JOINT_NAMES, [f'{math.degrees(r):.1f} deg' for r in arm_rad]))}")
        print(f"#   Output -> {pose_dir}")
        print(f"{'#' * 64}\n")

        results: dict = defaultdict(dict)

        for ji, jname in enumerate(ARM_JOINT_NAMES):
            arm_idx = arm_joint_ids[ji]
            print(f"--- {jname} ({ji + 1}/{NUM_ARM_JOINTS}) ---")

            _reset_robot(sim, robot, home_pos, sim_dt, settle_steps)
            _verify_reset(robot, arm_joint_ids, arm_rad)

            for step_deg in STEP_SIZES_DEG:
                times, positions, baseline_rad, target_rad = _record_step(
                    sim, robot, arm_idx, step_deg, sim_dt, record_steps
                )
                _log_step_result(jname, step_deg, baseline_rad, target_rad, positions)

                results[ji][step_deg] = {
                    "times": times,
                    "positions": positions,
                    "baseline_rad": baseline_rad,
                }
                _reset_robot(sim, robot, home_pos, sim_dt, settle_steps)

        print(f"\nGenerating plots for '{pose_label}'...")
        _generate_plots(results, pose_dir, sim_dt, pose_label, act_info)
        print(f"Plots saved to {pose_dir}")

    print(f"\nAll tests complete.  Results in {output_dir}")


def _verify_reset(robot: Articulation, arm_joint_ids: list[int], arm_rad: list[float]) -> None:
    """Print reset-position verification and warn if error exceeds threshold."""
    reset_pos = wp.to_torch(robot.data.joint_pos)[0]
    actual_deg = [math.degrees(reset_pos[aid].item()) for aid in arm_joint_ids]
    expected_deg = [math.degrees(r) for r in arm_rad]
    max_err = max(abs(a - e) for a, e in zip(actual_deg, expected_deg))

    summary = dict(zip(ARM_JOINT_NAMES, [f"{d:.3f}" for d in actual_deg]))
    print(f"  After reset -> {summary}")
    if max_err > 0.1:
        print(f"  WARNING: max reset error = {max_err:.4f} deg")
    else:
        print(f"  Reset OK  (max error {max_err:.4f} deg)")


def _record_step(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    arm_idx: int,
    step_deg: float,
    sim_dt: float,
    record_steps: int,
) -> tuple[list[float], list[float], float, float]:
    """Record a baseline period, then apply the step and record the transient.

    Matches the real-robot protocol: recording starts at t=0, the step command
    is applied after :data:`BASELINE_DURATION` seconds, and recording continues
    for *record_steps* physics steps after the step.

    Returns:
        ``(times, positions, baseline_rad, target_rad)``
    """
    step_rad = math.radians(step_deg)

    baseline_pos = wp.to_torch(robot.data.joint_pos).clone()
    baseline_rad = baseline_pos[0, arm_idx].item()
    target_rad = baseline_rad + step_rad

    target_pos = baseline_pos.clone()
    target_pos[0, arm_idx] = target_rad

    baseline_steps = int(BASELINE_DURATION / sim_dt)

    times: list[float] = []
    positions: list[float] = []
    total_step = 0

    for _ in range(baseline_steps):
        robot.set_joint_position_target_index(target=baseline_pos)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim_dt)

        total_step += 1
        times.append(total_step * sim_dt)
        positions.append(wp.to_torch(robot.data.joint_pos)[0, arm_idx].item())

    for _ in range(record_steps):
        robot.set_joint_position_target_index(target=target_pos)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim_dt)

        total_step += 1
        times.append(total_step * sim_dt)
        positions.append(wp.to_torch(robot.data.joint_pos)[0, arm_idx].item())

    return times, positions, baseline_rad, target_rad


def _log_step_result(
    jname: str,
    step_deg: float,
    baseline_rad: float,
    target_rad: float,
    positions: list[float],
) -> None:
    """Print a one-line summary of the step result."""
    final_deg = math.degrees(positions[-1])
    target_deg = math.degrees(target_rad)
    baseline_deg = math.degrees(baseline_rad)
    peak = math.degrees(max(positions, key=lambda p: abs(p - baseline_rad)) - baseline_rad)
    err = abs(final_deg - target_deg)
    print(
        f"  Step {step_deg:+.0f} deg on {jname}: "
        f"{baseline_deg:.2f} -> {target_deg:.2f} deg | "
        f"final={final_deg:.3f}  err={err:.4f}  peak={peak:.3f}"
    )


##
# Plotting
##

_STEP_COLORS: dict[float, str] = {
    -1.0: "tab:blue",
    1.0: "tab:orange",
    -5.0: "tab:green",
    5.0: "tab:red",
}


def _generate_plots(
    results: dict,
    output_dir: str,
    sim_dt: float,
    pose_label: str,
    act_info: dict[str, dict],
) -> None:
    """Generate per-joint and overview plots."""
    suffix = f" [{pose_label}]" if pose_label else ""
    _plot_per_joint(results, output_dir, sim_dt, suffix)
    _plot_overview(results, output_dir, sim_dt, suffix, act_info)


def _plot_per_joint(results: dict, output_dir: str, sim_dt: float, suffix: str) -> None:
    for ji in range(NUM_ARM_JOINTS):
        jname = f"joint{ji + 1}"
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"{jname} Step Response (Sim, dt={sim_dt * 1e3:.2f} ms){suffix}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Relative Position (deg)")
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(x=BASELINE_DURATION, color="black", linewidth=0.8, linestyle="--", alpha=0.4, label="step applied")

        for step_deg in STEP_SIZES_DEG:
            data = results[ji].get(step_deg)
            if data is None:
                continue
            rel_deg = [math.degrees(p - data["baseline_rad"]) for p in data["positions"]]
            ax.plot(data["times"], rel_deg, label=f"{step_deg:+.0f} deg", color=_STEP_COLORS[step_deg], linewidth=1.5)
            ax.axhline(y=step_deg, color=_STEP_COLORS[step_deg], linewidth=0.8, linestyle=":", alpha=0.5)

        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

        path = os.path.join(output_dir, f"{jname}_step_response_sim.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


def _plot_overview(
    results: dict,
    output_dir: str,
    sim_dt: float,
    suffix: str,
    act_info: dict[str, dict],
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
    fig.suptitle(f"Step Response Overview (Sim, dt={sim_dt * 1e3:.2f} ms){suffix}", fontsize=14)
    axes_flat = axes.flatten()

    for ji in range(NUM_ARM_JOINTS):
        ax = axes_flat[ji]
        ax.set_title(f"joint{ji + 1}")
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(x=BASELINE_DURATION, color="black", linewidth=0.6, linestyle="--", alpha=0.3)

        for step_deg in STEP_SIZES_DEG:
            data = results[ji].get(step_deg)
            if data is None:
                continue
            rel_deg = [math.degrees(p - data["baseline_rad"]) for p in data["positions"]]
            ax.plot(data["times"], rel_deg, label=f"{step_deg:+.0f}", color=_STEP_COLORS[step_deg], linewidth=1.2)
            ax.axhline(y=step_deg, color=_STEP_COLORS[step_deg], linewidth=0.6, linestyle=":", alpha=0.4)

        ax.grid(True, alpha=0.3)
        ax.set_ylabel("deg")
        if ji == 0:
            ax.legend(loc="best", fontsize=8)

    # Info panel in the unused 8th subplot
    info_ax = axes_flat[7]
    info_ax.axis("off")
    first = next(iter(act_info.values()))
    lines = [
        f"Actuator: {first['type']}",
        f"dt={TRAINING_DT * 1e3:.3f} ms  dec={TRAINING_DECIMATION}",
        "",
    ]
    for jname in ARM_JOINT_NAMES:
        a = act_info[jname]
        lines.append(f"{jname}: Kp={a['stiffness']:.0f}  Kd={a['damping']:.0f}")
    info_ax.text(
        0.1,
        0.5,
        "\n".join(lines),
        transform=info_ax.transAxes,
        fontsize=9,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    for ax in axes_flat[5:7]:
        ax.set_xlabel("Time (s)")

    path = os.path.join(output_dir, "overview_step_response_sim.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overview: {path}")


##
# Entry point
##


def main() -> None:
    """Set up the simulation, run the step-response test, and save plots."""
    sim_dt = args_cli.sim_dt if args_cli.sim_dt is not None else TRAINING_DT

    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 2.0, 2.0], [0.0, 0.0, 0.5])

    scene_entities = design_scene()
    robot = scene_entities["robot"]

    sim.reset()
    print("[INFO]: Setup complete – starting step-response test ...")

    run_step_response(
        sim,
        robot,
        output_dir=args_cli.output_dir,
        sim_dt=sim_dt,
        record_dur=args_cli.record_duration,
        settle_dur=args_cli.settle_duration,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
