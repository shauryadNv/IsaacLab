# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Class-based event terms specific to the gear assembly manipulation environments."""

from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv


class randomize_gear_type(ManagerTermBase):
    """Randomize and manage the gear type being used for each environment.

    This class stores the current gear type for each environment and provides a mapping
    from gear type names to indices. It serves as the central manager for gear type state
    that other MDP terms depend on.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the gear type randomization term.

        Args:
            cfg: Event term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Extract gear types from config (required parameter)
        if "gear_types" not in cfg.params:
            raise ValueError("'gear_types' parameter is required in randomize_gear_type configuration")
        self.gear_types: list[str] = cfg.params["gear_types"]

        # Create gear type mapping (shared across all terms)
        self.gear_type_map = {"gear_small": 0, "gear_medium": 1, "gear_large": 2}

        # Store current gear type for each environment (as list for easy access)
        # Initialize all to first gear type in the list
        self._current_gear_type = [self.gear_types[0]] * env.num_envs

        # Store current gear type indices as tensor for efficient vectorized access
        # Initialize all to first gear type index
        first_gear_idx = self.gear_type_map[self.gear_types[0]]
        self._current_gear_type_indices = torch.full(
            (env.num_envs,), first_gear_idx, device=env.device, dtype=torch.long
        )

        # Store reference on environment for other terms to access
        env._gear_type_manager = self

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        gear_types: list[str] = ["gear_small", "gear_medium", "gear_large"],
    ):
        """Randomize the gear type for specified environments.

        Args:
            env: The environment containing the assets
            env_ids: Environment IDs to randomize
            gear_types: List of available gear types to choose from
        """
        # Randomly select gear type for each environment
        # Use the parameter passed to __call__ (not self.gear_types) to allow runtime overrides
        for env_id in env_ids.tolist():
            chosen_gear = random.choice(gear_types)
            self._current_gear_type[env_id] = chosen_gear
            self._current_gear_type_indices[env_id] = self.gear_type_map[chosen_gear]

    def get_gear_type(self, env_id: int) -> str:
        """Get the current gear type for a specific environment."""
        return self._current_gear_type[env_id]

    def get_all_gear_types(self) -> list[str]:
        """Get current gear types for all environments."""
        return self._current_gear_type

    def get_all_gear_type_indices(self) -> torch.Tensor:
        """Get current gear type indices for all environments as a tensor.

        Returns:
            Tensor of shape (num_envs,) with gear type indices (0=small, 1=medium, 2=large)
        """
        return self._current_gear_type_indices


@dataclass
class _CuroboIKCacheEntry:
    solver: Any
    goal_buffer: Any | None = None
    seed_buffer: torch.Tensor | None = None


class set_robot_to_grasp_pose(ManagerTermBase):
    """Drop-in reset event term that uses cuRobo batched IK for grasp-pose initialization."""

    _solver_cache: dict[tuple[Any, ...], _CuroboIKCacheEntry] = {}

    def __init__(self, cfg: EventTermCfg, env):
        super().__init__(cfg, env)
        self.robot_asset_cfg: SceneEntityCfg = cfg.params.get("robot_asset_cfg", SceneEntityCfg("robot"))
        self.robot_asset = env.scene[self.robot_asset_cfg.name]

        for key in ["end_effector_body_name", "num_arm_joints", "grasp_rot_offset", "gripper_joint_setter_func", "gear_offsets_grasp"]:
            if key not in cfg.params:
                raise ValueError(f"'{key}' parameter is required for CuroboSetRobotToGraspPose")

        self.end_effector_body_name = cfg.params["end_effector_body_name"]
        self.num_arm_joints = cfg.params["num_arm_joints"]
        self.gripper_joint_setter_func = cfg.params["gripper_joint_setter_func"]
        self.hand_grasp_width = env.cfg.hand_grasp_width
        self.hand_close_width = env.cfg.hand_close_width

        gear_offsets_grasp = cfg.params["gear_offsets_grasp"]
        self.gear_grasp_offsets_stacked = torch.stack(
            [
                torch.tensor(gear_offsets_grasp[k], device=env.device, dtype=torch.float32)
                for k in ["gear_small", "gear_medium", "gear_large"]
            ],
            dim=0,
        )
        self.grasp_rot_offset_tensor = (
            torch.tensor(cfg.params["grasp_rot_offset"], device=env.device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(env.num_envs, 1)
        )
        self.gear_type_indices = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.local_env_indices = torch.arange(env.num_envs, device=env.device)
        self.gear_grasp_offsets_buffer = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
        self.full_target_pos_buffer = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
        self.full_target_quat_buffer = torch.zeros((env.num_envs, 4), device=env.device, dtype=torch.float32)
        self.full_target_quat_buffer[:, 0] = 1.0
        self.full_targets_initialized = False
        self.full_targets_dirty = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        self.gear_type_strings = (
            tuple(env._gear_type_manager.get_all_gear_types()) if hasattr(env, "_gear_type_manager") else None
        )
        self.gear_type_string_to_index = {"gear_small": 0, "gear_medium": 1, "gear_large": 2}
        self.finger_width_grasp_tensor = torch.zeros((3, 2), device=env.device, dtype=torch.float32)
        self.finger_width_close_tensor = torch.zeros((3, 2), device=env.device, dtype=torch.float32)
        for gear_key, gear_idx in self.gear_type_string_to_index.items():
            self.finger_width_grasp_tensor[gear_idx] = torch.tensor(
                env.cfg.hand_grasp_width[gear_key], device=env.device, dtype=torch.float32
            )
            self.finger_width_close_tensor[gear_idx] = torch.tensor(
                env.cfg.hand_close_width[gear_key], device=env.device, dtype=torch.float32
            )

        all_joints, _ = self.robot_asset.find_joints([".*"])
        self.all_joints = all_joints
        self.finger_joints = all_joints[self.num_arm_joints :]

        import isaaclab.utils.math as math_utils
        from curobo.types.base import TensorDeviceType
        from curobo.util_file import get_robot_configs_path, join_path, load_yaml
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

        self._math_utils = math_utils
        self._curobo_pose_cls = __import__("curobo.types.math", fromlist=["Pose"]).Pose
        self.num_seeds = int(cfg.params.get("curobo_num_seeds", 1))
        self.newton_iters = int(cfg.params.get("curobo_newton_iters", 1))
        self.full_batch = bool(cfg.params.get("curobo_full_batch", True))
        self.use_cuda_graph = bool(cfg.params.get("curobo_use_cuda_graph", True))
        self.robot_cfg_name = cfg.params.get("curobo_robot_cfg", "ur10e.yml")

        tensor_args = TensorDeviceType(device=torch.device(env.device), dtype=torch.float32)
        robot_cfg = load_yaml(join_path(get_robot_configs_path(), self.robot_cfg_name))["robot_cfg"]
        cache_key = (
            self.robot_cfg_name,
            str(env.device),
            self.num_seeds,
            int(cfg.params.get("curobo_grad_iters", 3)),
            self.use_cuda_graph,
            self.full_batch,
            int(env.num_envs if self.full_batch else -1),
        )
        cache_entry = self._solver_cache.get(cache_key)
        if cache_entry is None:
            ik_cfg = IKSolverConfig.load_from_robot_config(
                robot_cfg,
                None,
                tensor_args=tensor_args,
                num_seeds=self.num_seeds,
                position_threshold=float(cfg.params.get("pos_threshold", 0.005)),
                rotation_threshold=float(cfg.params.get("rot_threshold", 0.05)),
                use_cuda_graph=self.use_cuda_graph,
                self_collision_check=False,
                self_collision_opt=False,
                use_particle_opt=False,
                grad_iters=int(cfg.params.get("curobo_grad_iters", 3)),
                collision_checker_type=None,
                sync_cuda_time=False,
                regularization=True,
            )
            solver = IKSolver(ik_cfg)
            goal_pos = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
            goal_quat = torch.zeros((env.num_envs, 4), device=env.device, dtype=torch.float32)
            goal_quat[:, 0] = 1.0
            seed_buffer = (
                self.robot_asset.data.default_joint_pos.torch[:, : self.num_arm_joints]
                .clone()
                .unsqueeze(0)
                .contiguous()
            )
            goal_buffer = self._curobo_pose_cls(position=goal_pos, quaternion=goal_quat)
            cache_entry = _CuroboIKCacheEntry(solver=solver, goal_buffer=goal_buffer, seed_buffer=seed_buffer)
            self._solver_cache[cache_key] = cache_entry
        self._cache_entry = cache_entry
        self.solver = cache_entry.solver

        # Warm the fixed full-env batch so reset calls with tiny env_ids can reuse one CUDA graph shape.
        if self.use_cuda_graph and self.full_batch and cache_entry.goal_buffer is not None and cache_entry.seed_buffer is not None:
            try:
                self.solver.solve_batch(
                    cache_entry.goal_buffer,
                    seed_config=cache_entry.seed_buffer,
                    num_seeds=self.num_seeds,
                    return_seeds=1,
                    use_nn_seed=False,
                    newton_iters=self.newton_iters,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception as exc:
                print(f"[PROFILE_CUROBO_IK_WARMUP_FAILED] {exc!r}")

    def _compute_target_pose(self, env, env_ids, pos_randomization_range):
        profiler = getattr(env, "_reset_fine_profiler", None)
        ctx = profiler.section if profiler is not None else None
        math_utils = self._math_utils
        num_reset_envs = len(env_ids)
        gear_type_indices = self.gear_type_indices[:num_reset_envs]
        local_env_indices = self.local_env_indices[:num_reset_envs]
        gear_grasp_offsets = self.gear_grasp_offsets_buffer[:num_reset_envs]
        with (ctx("reset_fine.curobo_compute_target.slice_buffers") if ctx else contextlib.nullcontext()):
            grasp_rot_offset_tensor = self.grasp_rot_offset_tensor[env_ids]

        with (ctx("reset_fine.curobo_compute_target.stack_gear_pose") if ctx else contextlib.nullcontext()):
            all_gear_pos = torch.stack(
                [
                    env.scene["factory_gear_small"].data.root_link_pos_w.torch,
                    env.scene["factory_gear_medium"].data.root_link_pos_w.torch,
                    env.scene["factory_gear_large"].data.root_link_pos_w.torch,
                ],
                dim=1,
            )[env_ids]
            all_gear_quat = torch.stack(
                [
                    env.scene["factory_gear_small"].data.root_link_quat_w.torch,
                    env.scene["factory_gear_medium"].data.root_link_quat_w.torch,
                    env.scene["factory_gear_large"].data.root_link_quat_w.torch,
                ],
                dim=1,
            )[env_ids]
        with (ctx("reset_fine.curobo_compute_target.select_gear_type") if ctx else contextlib.nullcontext()):
            all_gear_type_indices = env._gear_type_manager.get_all_gear_type_indices()
            gear_type_indices[:] = all_gear_type_indices[env_ids]
            grasp_object_pos_world = all_gear_pos[local_env_indices, gear_type_indices]
            grasp_object_quat_xyzw = all_gear_quat[local_env_indices, gear_type_indices]
        with (ctx("reset_fine.curobo_compute_target.rotation_offsets") if ctx else contextlib.nullcontext()):
            grasp_object_quat_xyzw = math_utils.quat_mul(grasp_object_quat_xyzw, grasp_rot_offset_tensor)
            gear_grasp_offsets[:] = self.gear_grasp_offsets_stacked[gear_type_indices]
            if pos_randomization_range is not None:
                # Matches the manager reset term behavior: sample one grasp-offset perturbation per reset call,
                # not per environment.
                range_list_pos = [pos_randomization_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
                ranges_pos = torch.tensor(range_list_pos, device=env.device)
                rand_pos_offsets = math_utils.sample_uniform(
                    ranges_pos[:, 0], ranges_pos[:, 1], (1, 3), device=env.device
                )
                gear_grasp_offsets = gear_grasp_offsets + rand_pos_offsets
            grasp_object_pos_world = grasp_object_pos_world + math_utils.quat_apply(
                grasp_object_quat_xyzw, gear_grasp_offsets
            )
        with (ctx("reset_fine.curobo_compute_target.quat_reorder_contiguous") if ctx else contextlib.nullcontext()):
            grasp_object_quat_wxyz = torch.cat(
                [grasp_object_quat_xyzw[:, 3:4], grasp_object_quat_xyzw[:, 0:3]], dim=-1
            ).contiguous()
            return grasp_object_pos_world.contiguous(), grasp_object_quat_wxyz

    def __call__(
        self,
        env,
        env_ids: torch.Tensor,
        robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        pos_threshold: float = 1e-6,
        rot_threshold: float = 1e-6,
        max_iterations: int = 50,
        pos_randomization_range: dict | None = None,
        gear_offsets_grasp: dict | None = None,
        end_effector_body_name: str | None = None,
        num_arm_joints: int | None = None,
        grasp_rot_offset: list | None = None,
        gripper_joint_setter_func: Callable | None = None,
        curobo_robot_cfg: str | None = None,
        curobo_num_seeds: int | None = None,
        curobo_grad_iters: int | None = None,
        curobo_newton_iters: int | None = None,
        curobo_full_batch: bool | None = None,
        curobo_use_cuda_graph: bool | None = None,
    ):
        profiler = getattr(env, "_reset_fine_profiler", None)
        ctx = profiler.section if profiler is not None else None
        if not hasattr(env, "_gear_type_manager"):
            raise RuntimeError("Gear type manager not initialized before cuRobo reset grasp solver")
        with (ctx("reset_fine.curobo.compute_target_pose") if ctx else contextlib.nullcontext()):
            target_pos, target_quat_wxyz = self._compute_target_pose(env, env_ids, pos_randomization_range)

        if self.full_batch:
            with (ctx("reset_fine.curobo.full_batch_prepare") if ctx else contextlib.nullcontext()):
                pose = self._cache_entry.goal_buffer
                seed_config = self._cache_entry.seed_buffer
                if pose is None or seed_config is None:
                    raise RuntimeError("cuRobo full-batch cache was not initialized")
                # Keep non-reset rows at their last reachable reset targets instead of re-solving arbitrary
                # stale/identity goals. This preserves fixed CUDA-graph batch size without rebuilding all
                # targets every tiny reset call.
                if not self.full_targets_initialized:
                    full_env_ids = torch.arange(env.num_envs, device=env.device)
                    full_target_pos, full_target_quat = self._compute_target_pose(
                        env, full_env_ids, pos_randomization_range
                    )
                    self.full_target_pos_buffer.copy_(full_target_pos)
                    self.full_target_quat_buffer.copy_(full_target_quat)
                    self.full_targets_initialized = True
                    self.full_targets_dirty.zero_()
                elif torch.any(self.full_targets_dirty[env_ids]):
                    dirty_env_ids = torch.nonzero(self.full_targets_dirty, as_tuple=False).flatten()
                    dirty_target_pos, dirty_target_quat = self._compute_target_pose(
                        env, dirty_env_ids, pos_randomization_range
                    )
                    self.full_target_pos_buffer[dirty_env_ids] = dirty_target_pos
                    self.full_target_quat_buffer[dirty_env_ids] = dirty_target_quat
                    self.full_targets_dirty.zero_()
                self.full_target_pos_buffer[env_ids] = target_pos
                self.full_target_quat_buffer[env_ids] = target_quat_wxyz
                pose.position.copy_(self.full_target_pos_buffer)
                pose.quaternion.copy_(self.full_target_quat_buffer)
                seed_config[0].copy_(self.robot_asset.data.joint_pos.torch[:, : self.num_arm_joints])
        else:
            with (ctx("reset_fine.curobo.variable_batch_prepare") if ctx else contextlib.nullcontext()):
                seed_joint_pos = self.robot_asset.data.joint_pos.torch[env_ids, : self.num_arm_joints].clone()
                seed_config = seed_joint_pos.unsqueeze(0)
                pose = self._curobo_pose_cls(position=target_pos, quaternion=target_quat_wxyz)

        with (ctx("reset_fine.curobo.solve_batch") if ctx else contextlib.nullcontext()):
            result = self.solver.solve_batch(
                pose,
                seed_config=seed_config,
                num_seeds=self.num_seeds,
                return_seeds=1,
                use_nn_seed=False,
                newton_iters=self.newton_iters,
            )
        with (ctx("reset_fine.curobo.extract_solution_success") if ctx else contextlib.nullcontext()):
            solution = result.solution[:, 0, :] if result.solution.ndim == 3 else result.solution
            if self.full_batch:
                arm_solution = solution[env_ids]
                success = result.success[env_ids].flatten()
            else:
                arm_solution = solution
                success = result.success.flatten()
            if not torch.all(success):
                # Keep the seed joints for failed solves rather than poisoning reset with NaNs/large jumps.
                fallback = self.robot_asset.data.joint_pos.torch[env_ids, : self.num_arm_joints]
                arm_solution = torch.where(success.view(-1, 1), arm_solution, fallback)

        with (ctx("reset_fine.curobo.build_joint_state") if ctx else contextlib.nullcontext()):
            joint_pos = self.robot_asset.data.joint_pos.torch[env_ids].clone()
            joint_pos[:, : self.num_arm_joints] = arm_solution
            joint_vel = torch.zeros_like(joint_pos)

        with (ctx("reset_fine.curobo.set_gripper_grasp") if ctx else contextlib.nullcontext()):
            gear_type_indices = env._gear_type_manager.get_all_gear_type_indices()[env_ids]
            if len(self.finger_joints) == 2:
                joint_pos[:, self.num_arm_joints :] = self.finger_width_grasp_tensor[gear_type_indices]
            else:
                all_gear_types = self.gear_type_strings or env._gear_type_manager.get_all_gear_types()
                for row_idx, env_id in enumerate(env_ids.tolist()):
                    gear_key = all_gear_types[env_id]
                    self.gripper_joint_setter_func(
                        joint_pos, [row_idx], self.finger_joints, self.hand_grasp_width[gear_key]
                    )
        with (ctx("reset_fine.curobo.set_joint_position_target_grasp") if ctx else contextlib.nullcontext()):
            self.robot_asset.set_joint_position_target_index(target=joint_pos, joint_ids=self.all_joints, env_ids=env_ids)
        with (ctx("reset_fine.curobo.write_joint_position_to_sim") if ctx else contextlib.nullcontext()):
            self.robot_asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        with (ctx("reset_fine.curobo.write_joint_velocity_to_sim") if ctx else contextlib.nullcontext()):
            self.robot_asset.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        with (ctx("reset_fine.curobo.set_gripper_close") if ctx else contextlib.nullcontext()):
            if len(self.finger_joints) == 2:
                joint_pos[:, self.num_arm_joints :] = self.finger_width_close_tensor[gear_type_indices]
            else:
                all_gear_types = self.gear_type_strings or env._gear_type_manager.get_all_gear_types()
                for row_idx, env_id in enumerate(env_ids.tolist()):
                    gear_key = all_gear_types[env_id]
                    self.gripper_joint_setter_func(
                        joint_pos, [row_idx], self.finger_joints, self.hand_close_width[gear_key]
                    )
        with (ctx("reset_fine.curobo.set_joint_position_target_close") if ctx else contextlib.nullcontext()):
            self.robot_asset.set_joint_position_target_index(target=joint_pos, joint_ids=self.all_joints, env_ids=env_ids)
        if self.full_batch:
            with (ctx("reset_fine.curobo.mark_full_targets_dirty") if ctx else contextlib.nullcontext()):
                # These envs may now be moving under policy control, so recompute their non-reset CUDA-graph goals
                # only if/when they are needed again as padding rows.
                self.full_targets_dirty[env_ids] = True

class randomize_gears_and_base_pose(ManagerTermBase):
    """Randomize both the gear base pose and individual gear poses.

    This class-based term pre-caches all tensors needed for randomization.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the randomize gears and base pose term.

        Args:
            cfg: Event term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Pre-allocate gear type mapping and indices
        self.gear_type_map = {"gear_small": 0, "gear_medium": 1, "gear_large": 2}
        self.gear_type_indices = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

        # Cache asset names
        self.gear_asset_names = ["factory_gear_small", "factory_gear_medium", "factory_gear_large"]
        self.base_asset_name = "factory_gear_base"

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        pose_range: dict = {},
        velocity_range: dict = {},
        gear_pos_range: dict = {},
    ):
        """Randomize gear base and gear poses.

        Args:
            env: Environment instance
            env_ids: Environment IDs to randomize
            pose_range: Pose randomization range for base and all gears
            velocity_range: Velocity randomization range
            gear_pos_range: Additional position randomization for selected gear only
        """
        if not hasattr(env, "_gear_type_manager"):
            raise RuntimeError(
                "Gear type manager not initialized. Ensure randomize_gear_type event is configured "
                "in your environment's event configuration before this event term is used."
            )

        gear_type_manager: randomize_gear_type = env._gear_type_manager
        device = env.device

        # Shared pose samples for all assets
        pose_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        range_list_pose = [pose_range.get(key, (0.0, 0.0)) for key in pose_keys]
        ranges_pose = torch.tensor(range_list_pose, device=device)
        rand_pose_samples = math_utils.sample_uniform(
            ranges_pose[:, 0], ranges_pose[:, 1], (len(env_ids), 6), device=device
        )

        orientations_delta = math_utils.quat_from_euler_xyz(
            rand_pose_samples[:, 3], rand_pose_samples[:, 4], rand_pose_samples[:, 5]
        )

        # Shared velocity samples
        range_list_vel = [velocity_range.get(key, (0.0, 0.0)) for key in pose_keys]
        ranges_vel = torch.tensor(range_list_vel, device=device)
        rand_vel_samples = math_utils.sample_uniform(
            ranges_vel[:, 0], ranges_vel[:, 1], (len(env_ids), 6), device=device
        )

        # Prepare poses for all assets
        positions_by_asset = {}
        orientations_by_asset = {}
        velocities_by_asset = {}

        asset_names_to_process = [self.base_asset_name] + self.gear_asset_names
        for asset_name in asset_names_to_process:
            asset: RigidObject | Articulation = env.scene[asset_name]
            default_root_pose = asset.data.default_root_pose.torch[env_ids].clone()
            default_root_vel = asset.data.default_root_vel.torch[env_ids].clone()
            positions = default_root_pose[:, 0:3] + env.scene.env_origins[env_ids] + rand_pose_samples[:, 0:3]
            orientations = math_utils.quat_mul(default_root_pose[:, 3:7], orientations_delta)
            velocities = default_root_vel + rand_vel_samples
            positions_by_asset[asset_name] = positions
            orientations_by_asset[asset_name] = orientations
            velocities_by_asset[asset_name] = velocities

        # Per-env gear offset (gear_pos_range) applied only to selected gear
        range_list_gear = [gear_pos_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        ranges_gear = torch.tensor(range_list_gear, device=device)
        rand_gear_offsets = math_utils.sample_uniform(
            ranges_gear[:, 0], ranges_gear[:, 1], (len(env_ids), 3), device=device
        )

        # Get gear type indices directly as tensor
        num_reset_envs = len(env_ids)
        gear_type_indices = self.gear_type_indices[:num_reset_envs]
        all_gear_type_indices = gear_type_manager.get_all_gear_type_indices()
        gear_type_indices[:] = all_gear_type_indices[env_ids]

        # Apply offsets using vectorized operations with masks
        for gear_idx, asset_name in enumerate(self.gear_asset_names):
            if asset_name in positions_by_asset:
                mask = gear_type_indices == gear_idx
                positions_by_asset[asset_name][mask] = positions_by_asset[asset_name][mask] + rand_gear_offsets[mask]

        # Write to sim
        for asset_name in positions_by_asset.keys():
            asset = env.scene[asset_name]
            positions = positions_by_asset[asset_name]
            orientations = orientations_by_asset[asset_name]
            velocities = velocities_by_asset[asset_name]
            asset.write_root_pose_to_sim_index(root_pose=torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
            asset.write_root_velocity_to_sim_index(root_velocity=velocities, env_ids=env_ids)
