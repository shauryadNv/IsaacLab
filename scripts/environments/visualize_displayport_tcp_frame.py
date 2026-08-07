# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize the DisplayPort policy's TCP observation beside the controlled flange frame."""

import argparse
import contextlib
import math
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

from isaaclab import sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG, VisualizationMarkersCfg

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

_DEFAULT_TASK = "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Calibrated-DR-Newton-IK-TcpObs"
_HEADLESS_STEPS = 100

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default=_DEFAULT_TASK, help="Task containing the TCP actor observation.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to visualize.")
parser.add_argument("--max_steps", type=int, default=None, help="Stop after this many policy steps.")
parser.add_argument("--print_interval", type=int, default=30, help="Policy steps between pose printouts.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def _observation_term_slice(env, group_name: str, term_name: str) -> slice:
    """Return the flattened slice occupied by an observation term."""
    manager = env.observation_manager
    if not manager.group_obs_concatenate[group_name]:
        raise ValueError(f"Observation group '{group_name}' must concatenate its terms.")

    start = 0
    for name, shape in zip(manager.active_terms[group_name], manager.group_obs_term_dim[group_name]):
        width = math.prod(shape)
        if name == term_name:
            return slice(start, start + width)
        start += width
    raise ValueError(f"Observation term '{term_name}' is not active in group '{group_name}'.")


def _frame_marker(prim_path: str, scale: float) -> VisualizationMarkers:
    """Create an axes marker at the requested scale."""
    cfg = FRAME_MARKER_CFG.copy()
    cfg.prim_path = prim_path
    cfg.markers["frame"].scale = (scale, scale, scale)
    return VisualizationMarkers(cfg)


def _origin_marker(prim_path: str, color: tuple[float, float, float], radius: float) -> VisualizationMarkers:
    """Create a colored origin marker."""
    cfg = VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "origin": sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
        },
    )
    return VisualizationMarkers(cfg)


def main():
    """Run a zero-action rollout and visualize the observed TCP and controlled flange frames."""
    env_cfg, _ = resolve_task_config(args_cli.task, "")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device

        env = gym.make(args_cli.task, cfg=env_cfg)
        unwrapped = env.unwrapped
        flange_cfg = SceneEntityCfg("robot", body_names=["flange"])
        flange_cfg.resolve(unwrapped.scene)
        flange_body_id = flange_cfg.body_ids[0]
        tcp_slice = _observation_term_slice(unwrapped, "policy", "tcp_pose")

        flange_frame = _frame_marker("/Visuals/DisplayPortFrames/FlangeFrame", 0.05)
        tcp_frame = _frame_marker("/Visuals/DisplayPortFrames/TcpObservationFrame", 0.09)
        flange_origin = _origin_marker("/Visuals/DisplayPortFrames/FlangeOrigin", (0.1, 0.4, 1.0), 0.004)
        tcp_origin = _origin_marker("/Visuals/DisplayPortFrames/TcpObservationOrigin", (1.0, 0.1, 0.7), 0.006)

        observations, _ = env.reset()
        actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
        step = 0
        print("[INFO] Small axes / blue origin: controlled flange frame")
        print("[INFO] Large axes / magenta origin: policy TCP observation frame")

        while True:
            sim = unwrapped.sim
            if sim.visualizers and not any(v.is_running() and not v.is_closed for v in sim.visualizers):
                break
            if args_cli.max_steps is not None and step >= args_cli.max_steps:
                break
            if not sim.visualizers and args_cli.max_steps is None and step >= _HEADLESS_STEPS:
                break

            policy_obs = observations["policy"]
            tcp_pose = policy_obs[:, tcp_slice]
            tcp_pos_w = tcp_pose[:, :3] + unwrapped.scene.env_origins
            tcp_quat_w = tcp_pose[:, 3:7]
            flange_pose_w = unwrapped.scene["robot"].data.body_pose_w.torch[:, flange_body_id, :7]

            flange_frame.visualize(translations=flange_pose_w[:, :3], orientations=flange_pose_w[:, 3:7])
            tcp_frame.visualize(translations=tcp_pos_w, orientations=tcp_quat_w)
            flange_origin.visualize(translations=flange_pose_w[:, :3])
            tcp_origin.visualize(translations=tcp_pos_w)

            if step % args_cli.print_interval == 0:
                separation = torch.linalg.vector_norm(tcp_pos_w[0] - flange_pose_w[0, :3]).item()
                print(
                    f"[step {step}] flange={flange_pose_w[0, :3].tolist()} "
                    f"tcp_obs={tcp_pos_w[0].tolist()} separation={separation:.6f} m"
                )

            with torch.inference_mode():
                observations = env.step(actions)[0]
            step += 1

        env.close()


if __name__ == "__main__":
    main()
