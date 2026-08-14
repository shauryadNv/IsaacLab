# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Export the DisplayPort task-space policy with a Deploy InputBuilder I/O contract.

The policy graph is exported as:

    [eef_pos, eef_rot_6d, socket_kp_pos, socket_kp_rot_6d] -> arm_action

``arm_action`` is the clipped and scaled 6D Cartesian delta consumed by the
task-space bridge. The actor numerics and tensor ordering stay identical to the
trained RSL-RL policy path; only the external tensor names/metadata are made
stable for Isaac ROS Deploy.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path

from isaaclab.app import launch_simulation


_EXPORT_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "reinforcement_learning" / "leapp" / "rsl_rl"
if str(_EXPORT_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPORT_SCRIPT_DIR))

import export as rsl_rl_export  # isort: skip


DEFAULT_TASKSPACE_EXPORT_MODEL_NAME = "DisplayPortTaskSpace"


def _policy_obs(obs):
    """Return the actor observation tensor from an RSL-RL TensorDict."""
    if "policy" not in obs.keys():
        raise KeyError(f"Expected a 'policy' observation group, got keys: {list(obs.keys())}")
    policy_obs = obs["policy"]
    if policy_obs.shape[-1] != 18:
        raise ValueError(f"Expected 18D task-space actor observation, got shape {tuple(policy_obs.shape)}")
    return policy_obs


def _input_tensor(
    task_name: str,
    name: str,
    tensor,
    element_names: list[str],
    *,
    source: str,
    kind: str,
):
    """Annotate one Deploy-facing policy input tensor."""
    from leapp import annotate
    from leapp.utils.tensor_description import TensorSemantics

    return annotate.input_tensors(
        task_name,
        TensorSemantics(
            name=name,
            ref=tensor,
            kind=kind,
            element_names=[element_names],
            extra={"source": source},
        ),
    )


def _export_action(task_name: str, tensor, export_method: str):
    """Annotate the deploy-facing processed task-space action."""
    from leapp import annotate
    from leapp.utils.tensor_description import TensorSemantics

    annotate.output_tensors(
        task_name,
        TensorSemantics(
            name="arm_action",
            ref=tensor,
            kind="target/body/pose_relative",
            element_names=[
                [
                    "delta_x",
                    "delta_y",
                    "delta_z",
                    "delta_axis_angle_x",
                    "delta_axis_angle_y",
                    "delta_axis_angle_z",
                ]
            ],
            extra={
                "isaaclab_connection": "action:arm_action:pose_rel",
                "target_types": ["pose_rel"],
            },
        ),
        export_with=export_method,
    )


def _split_and_annotate_obs(task_name: str, policy_obs):
    """Expose the exact 18D trained observation as four Deploy inputs."""
    eef_pos = _input_tensor(
        task_name,
        "eef_pos",
        policy_obs[:, 0:3],
        ["x", "y", "z"],
        source="eef_pose_pos",
        kind="state/body/position",
    )
    eef_rot_6d = _input_tensor(
        task_name,
        "eef_rot_6d",
        policy_obs[:, 3:9],
        ["r00", "r01", "r02", "r10", "r11", "r12"],
        source="eef_pose_rot6d",
        kind="state/body/rotation_6d",
    )
    socket_kp_pos = _input_tensor(
        task_name,
        "socket_kp_pos",
        policy_obs[:, 9:12],
        ["x", "y", "z"],
        source="socket_kp_pose_pos",
        kind="state/body/position",
    )
    socket_kp_rot_6d = _input_tensor(
        task_name,
        "socket_kp_rot_6d",
        policy_obs[:, 12:18],
        ["r00", "r01", "r02", "r10", "r11", "r12"],
        source="socket_kp_pose_rot6d",
        kind="state/body/rotation_6d",
    )
    return rsl_rl_export.torch.cat([eef_pos, eef_rot_6d, socket_kp_pos, socket_kp_rot_6d], dim=-1)


def _task_space_action_scale(env_cfg, device, dtype):
    """Return [pos_scale]*3 + [rot_scale]*3 from the exact task config."""
    action_cfg = env_cfg.actions.arm_action
    scale_values = [float(action_cfg.position_scale)] * 3 + [float(action_cfg.orientation_scale)] * 3
    return rsl_rl_export.torch.tensor(scale_values, device=device, dtype=dtype).unsqueeze(0)


def _export_graph_name(args_cli) -> str:
    """Return the stable Deploy-facing LEAPP model name unless explicitly overridden."""
    return args_cli.export_task_name or DEFAULT_TASKSPACE_EXPORT_MODEL_NAME


def export_taskspace_policy(args_cli, env_cfg, agent_cfg, simulation_app=None) -> bool:
    """Export the task-space actor policy directly, with four observation inputs."""
    rsl_rl_export._load_runtime_dependencies()

    task_name = args_cli.task.split(":")[-1]
    checkpoint_task_name = task_name.replace("-Play", "")
    graph_name = _export_graph_name(args_cli)
    print(f"[INFO] Export graph/model name: {graph_name}")

    agent_cfg = rsl_rl_export.cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = rsl_rl_export.handle_deprecated_rsl_rl_cfg(agent_cfg, rsl_rl_export.installed_version)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading checkpoint search path from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = rsl_rl_export.get_published_pretrained_checkpoint("rsl_rl", checkpoint_task_name)
        if not resume_path:
            print(f"[INFO] No published checkpoint for task: {checkpoint_task_name}")
            return False
    elif args_cli.checkpoint:
        resume_path = rsl_rl_export.retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = rsl_rl_export.get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if not resume_path:
        print(f"[INFO] No checkpoint found for task: {checkpoint_task_name}")
        return False

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = None
    leapp_started = False
    try:
        env = rsl_rl_export.gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        if isinstance(env.unwrapped, rsl_rl_export.ManagerBasedRLEnv):
            obs_groups_cfg = getattr(agent_cfg, "obs_groups", None)
            if isinstance(obs_groups_cfg, Mapping):
                required_obs_groups = set(obs_groups_cfg.get("actor", ["policy"]))
            else:
                required_obs_groups = {"policy"}
            print(f"[INFO] Actor observation groups: {sorted(required_obs_groups)}")

        env = rsl_rl_export.RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = rsl_rl_export.OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = rsl_rl_export.DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        save_path = args_cli.export_save_path or log_dir
        rsl_rl_export.leapp.start(graph_name, save_path=save_path, max_cached_io=max(args_cli.validation_steps, 2))
        leapp_started = True

        obs = env.reset()[0]
        if simulation_app is not None:
            while not simulation_app.is_running():
                time.sleep(0.5)

        dtype = next(policy.parameters()).dtype
        action_scale = _task_space_action_scale(env.unwrapped.cfg, env.unwrapped.device, dtype)
        print(f"[INFO] Exporting processed task-space action with scale: {action_scale.flatten().tolist()}")

        for _ in range(max(args_cli.validation_steps, 2)):
            with rsl_rl_export.torch.inference_mode():
                policy_obs = _policy_obs(obs).to(dtype=dtype)
                actor_obs = _split_and_annotate_obs(graph_name, policy_obs)
                obs_for_policy = obs.clone()
                obs_for_policy["policy"] = actor_obs

                if rsl_rl_export.is_actor_recurrent_policy(policy):
                    actor_hidden = rsl_rl_export.ensure_actor_hidden_state_initialized(
                        policy,
                        batch_size=env.num_envs,
                        device=env.unwrapped.device,
                        dtype=dtype,
                    )
                    registered_state = rsl_rl_export.annotate.state_tensors(
                        graph_name,
                        rsl_rl_export.state_dict_from_actor_hidden(actor_hidden),
                    )
                    rsl_rl_export.set_actor_hidden_state(
                        policy,
                        rsl_rl_export.actor_hidden_from_registered(registered_state, actor_hidden),
                    )

                raw_action = policy(obs_for_policy)
                processed_action = rsl_rl_export.torch.clamp(raw_action, -1.0, 1.0) * action_scale

                if rsl_rl_export.is_actor_recurrent_policy(policy):
                    actor_hidden_after = rsl_rl_export.get_actor_hidden_state(policy)
                    rsl_rl_export.annotate.update_state(
                        graph_name,
                        rsl_rl_export.state_dict_from_actor_hidden(actor_hidden_after),
                    )

                _export_action(graph_name, processed_action, args_cli.export_method)

                # Keep the input samples changing without invoking env.step/action-manager behavior.
                obs = env.get_observations()

        rsl_rl_export.leapp.stop()
        leapp_started = False
        validate = args_cli.validation_steps > 0
        rsl_rl_export.leapp.compile_graph(visualize=not args_cli.disable_graph_visualization, validate=validate)
    finally:
        if leapp_started:
            with contextlib.suppress(Exception):
                rsl_rl_export.leapp.stop()
        if env is not None:
            env.close()

    return True


def run_export_with_hydra(args_cli, hydra_args: list[str]) -> bool:
    """Resolve Hydra task config and run direct task-space policy export."""
    from isaaclab_tasks.utils.hydra import hydra_task_config

    original_argv = sys.argv
    sys.argv = [sys.argv[0]] + hydra_args
    exported = False

    try:

        @hydra_task_config(args_cli.task, args_cli.agent)
        def _main(env_cfg, agent_cfg) -> None:
            nonlocal exported
            with launch_simulation(env_cfg, args_cli):
                exported = export_taskspace_policy(args_cli, env_cfg, agent_cfg)

        _main()
    finally:
        sys.argv = original_argv

    return exported


def main(argv: list[str] | None = None) -> bool:
    args_cli, hydra_args = rsl_rl_export.parse_export_args(argv)
    return run_export_with_hydra(args_cli, hydra_args)


if __name__ == "__main__":
    main()
