# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Class-based observation terms for manipulation deployment environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.leapp import (
    QUAT_XYZW_ELEMENT_NAMES,
    XYZ_ELEMENT_NAMES,
    InputKindEnum,
)

ROT6D_ELEMENT_NAMES: list[str] = ["r00", "r01", "r02", "r10", "r11", "r12"]
from isaaclab.utils.math import combine_frame_transforms, matrix_from_quat

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv

    from .events import randomize_gear_type


_LEAPP_TRACED_OBSERVATION_INPUTS = "_leapp_traced_observation_inputs"
_LEAPP_CONSUMED_OBSERVATION_INPUTS = "_leapp_consumed_observation_inputs"


def _tensor_data_to_torch(data) -> torch.Tensor:
    """Return a torch tensor view for Isaac Lab data stored as torch or Warp-backed data."""
    return data.torch if hasattr(data, "torch") else wp.to_torch(data)


def _selected_joint_names(asset, joint_ids) -> list[str] | None:
    """Return joint names selected by the observation config."""
    joint_names = getattr(asset, "joint_names", None)
    if joint_names is None:
        return None
    if joint_ids is None or joint_ids == slice(None):
        return list(joint_names)
    if isinstance(joint_ids, slice):
        return list(joint_names[joint_ids])
    if hasattr(joint_ids, "tolist"):
        joint_ids = joint_ids.tolist()
    return [joint_names[int(joint_id)] for joint_id in joint_ids]


def _is_leapp_export_env(env) -> bool:
    """Return whether the observation is running under the LEAPP export proxy."""
    return type(env).__name__ == "_EnvProxy"


def _leapp_real_env(env):
    """Return the wrapped Isaac Lab env when LEAPP passes an export proxy."""
    if _is_leapp_export_env(env):
        return object.__getattribute__(env, "_real_env")
    return env


def _set_leapp_traced_observation_input(env, name: str, tensor: torch.Tensor) -> None:
    """Store a traced observation tensor for later export-only reuse."""
    if not _is_leapp_export_env(env):
        return
    real_env = _leapp_real_env(env)
    traced_inputs = getattr(real_env, _LEAPP_TRACED_OBSERVATION_INPUTS, None)
    if traced_inputs is None:
        traced_inputs = {}
        setattr(real_env, _LEAPP_TRACED_OBSERVATION_INPUTS, traced_inputs)
    traced_inputs[name] = tensor
    getattr(real_env, _LEAPP_CONSUMED_OBSERVATION_INPUTS, set()).discard(name)


def _deploy_object_input_base_name(asset_name: str) -> str:
    """Return a stable deploy input base name for common socket/plug assets."""
    for prefix in ("dp_", "gb300_", "factory_"):
        if asset_name.startswith(prefix):
            return asset_name[len(prefix) :]
    return asset_name


def joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint positions for the configured joints, exposed as the LEAPP input boundary."""
    real_env = _leapp_real_env(env)
    asset = real_env.scene[asset_cfg.name]
    selected_joint_pos = _tensor_data_to_torch(asset.data.joint_pos)[:, asset_cfg.joint_ids]
    joint_names = _selected_joint_names(asset, asset_cfg.joint_ids)
    if _is_leapp_export_env(env):
        from leapp import annotate
        from leapp.utils.tensor_description import TensorSemantics

        selected_joint_pos = annotate.input_tensors(
            env.unwrapped.spec.id,
            TensorSemantics(
                name=f"{asset_cfg.name}_joint_pos",
                ref=selected_joint_pos,
                kind=InputKindEnum.JOINT_POSITION,
                element_names=joint_names,
                extra={"isaaclab_connection": f"state:{asset_cfg.name}:joint_pos"},
            ),
        )
        _set_leapp_traced_observation_input(env, f"{asset_cfg.name}_joint_pos", selected_joint_pos)
    return selected_joint_pos


def joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint velocities for the configured joints, exposed as the LEAPP input boundary."""
    real_env = _leapp_real_env(env)
    asset = real_env.scene[asset_cfg.name]
    selected_joint_vel = _tensor_data_to_torch(asset.data.joint_vel)[:, asset_cfg.joint_ids]
    joint_names = _selected_joint_names(asset, asset_cfg.joint_ids)
    if _is_leapp_export_env(env):
        from leapp import annotate
        from leapp.utils.tensor_description import TensorSemantics

        selected_joint_vel = annotate.input_tensors(
            env.unwrapped.spec.id,
            TensorSemantics(
                name=f"{asset_cfg.name}_joint_vel",
                ref=selected_joint_vel,
                kind=InputKindEnum.JOINT_VELOCITY,
                element_names=joint_names,
                extra={"isaaclab_connection": f"state:{asset_cfg.name}:joint_vel"},
            ),
        )
    return selected_joint_vel


class gear_shaft_pos_w(ManagerTermBase):
    """Gear shaft position in world frame with offset applied.

    This class-based term caches gear offset tensors and identity quaternions for efficient computation
    across all environments. It transforms the gear base position by the appropriate offset based on the
    active gear type in each environment.

    Args:
        asset_cfg: The asset configuration for the gear base. Defaults to SceneEntityCfg("factory_gear_base").
        gear_offsets: A dictionary mapping gear type names to their shaft offsets in the gear base frame.
            Required keys are "gear_small", "gear_medium", and "gear_large", each mapping to a 3D offset
            list [x, y, z]. This parameter is required and must be provided in the configuration.

    Returns:
        Gear shaft position tensor in the environment frame with shape (num_envs, 3).

    Raises:
        ValueError: If the 'gear_offsets' parameter is not provided in the configuration.
        TypeError: If the 'gear_offsets' parameter is not a dictionary.
        ValueError: If any of the required gear type keys are missing from 'gear_offsets'.
        RuntimeError: If the gear type manager is not initialized in the environment.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        """Initialize the gear shaft position observation term.

        Args:
            cfg: Observation term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Cache asset
        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("factory_gear_base"))
        self.asset: RigidObject = env.scene[self.asset_cfg.name]

        # Pre-cache gear offset tensors (required parameter)
        if "gear_offsets" not in cfg.params:
            raise ValueError(
                "'gear_offsets' parameter is required in gear_shaft_pos_w configuration. "
                "It should be a dict with keys 'gear_small', 'gear_medium', 'gear_large' mapping to [x, y, z] offsets."
            )
        gear_offsets = cfg.params["gear_offsets"]
        if not isinstance(gear_offsets, dict):
            raise TypeError(
                f"'gear_offsets' parameter must be a dict, got {type(gear_offsets).__name__}. "
                "It should have keys 'gear_small', 'gear_medium', 'gear_large' mapping to [x, y, z] offsets."
            )

        self.gear_offset_tensors = {}
        for gear_type in ["gear_small", "gear_medium", "gear_large"]:
            if gear_type not in gear_offsets:
                raise ValueError(
                    f"'{gear_type}' offset is required in 'gear_offsets' parameter. "
                    f"Found keys: {list(gear_offsets.keys())}"
                )
            self.gear_offset_tensors[gear_type] = torch.tensor(
                gear_offsets[gear_type], device=env.device, dtype=torch.float32
            )

        # Stack offset tensors for vectorized indexing (shape: 3, 3)
        # Index 0=small, 1=medium, 2=large
        self.gear_offsets_stacked = torch.stack(
            [
                self.gear_offset_tensors["gear_small"],
                self.gear_offset_tensors["gear_medium"],
                self.gear_offset_tensors["gear_large"],
            ],
            dim=0,
        )

        # Pre-allocate buffers
        self.offsets_buffer = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
        self.env_indices = torch.arange(env.num_envs, device=env.device)
        self.identity_quat = (
            torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=env.device, dtype=torch.float32)
            .repeat(env.num_envs, 1)
            .contiguous()
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("factory_gear_base"),
        gear_offsets: dict | None = None,
    ) -> torch.Tensor:
        """Compute gear shaft position in world frame.

        Args:
            env: Environment instance
            asset_cfg: Configuration of the gear base asset (unused, kept for compatibility)

        Returns:
            Gear shaft position tensor of shape (num_envs, 3)
        """
        # Check if gear type manager exists
        # During initialization (shape checking), the manager may not exist yet
        if not hasattr(env, "_gear_type_manager"):
            # Return default shape during initialization
            return torch.zeros(env.num_envs, 3, device=env.device)

        gear_type_manager: randomize_gear_type = env._gear_type_manager
        # Get gear type indices directly as tensor (no Python loops!)
        gear_type_indices = gear_type_manager.get_all_gear_type_indices()

        # Get base gear position and orientation
        base_pos = self.asset.data.root_pos_w.torch
        base_quat = self.asset.data.root_quat_w.torch

        # Update offsets using vectorized indexing
        self.offsets_buffer = self.gear_offsets_stacked[gear_type_indices]

        # Transform offsets
        shaft_pos, _ = combine_frame_transforms(base_pos, base_quat, self.offsets_buffer, self.identity_quat)

        return shaft_pos - env.scene.env_origins


class gear_shaft_quat_w(ManagerTermBase):
    """Gear shaft orientation in world frame.

    This class-based term returns the orientation of the gear base (which is the same as the gear shaft
    orientation). The quaternion is canonicalized to ensure the w component is positive, reducing
    observation variation for the policy.

    Args:
        asset_cfg: The asset configuration for the gear base. Defaults to SceneEntityCfg("factory_gear_base").

    Returns:
        Gear shaft orientation tensor as a quaternion (w, x, y, z) with shape (num_envs, 4).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        """Initialize the gear shaft orientation observation term.

        Args:
            cfg: Observation term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Cache asset
        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("factory_gear_base"))
        self.asset: RigidObject = env.scene[self.asset_cfg.name]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("factory_gear_base"),
    ) -> torch.Tensor:
        """Compute gear shaft orientation in world frame.

        Args:
            env: Environment instance
            asset_cfg: Configuration of the gear base asset (unused, kept for compatibility)

        Returns:
            Gear shaft orientation tensor of shape (num_envs, 4)
        """
        # Get base quaternion
        base_quat = self.asset.data.root_quat_w.torch

        # Ensure w component is positive (q and -q represent the same rotation)
        # Pick one canonical form to reduce observation variation seen by the policy
        w_negative = base_quat[:, 3] < 0
        positive_quat = torch.where(w_negative.unsqueeze(-1), -base_quat, base_quat)

        return positive_quat


class gear_pos_w(ManagerTermBase):
    """Gear position in world frame.

    This class-based term returns the position of the active gear in each environment. It uses
    vectorized indexing to efficiently select the correct gear position based on the gear type
    (small, medium, or large) active in each environment.

    Returns:
        Gear position tensor in the environment frame with shape (num_envs, 3).

    Raises:
        RuntimeError: If the gear type manager is not initialized in the environment.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        """Initialize the gear position observation term.

        Args:
            cfg: Observation term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Pre-allocate gear type mapping and indices
        self.gear_type_map = {"gear_small": 0, "gear_medium": 1, "gear_large": 2}
        self.gear_type_indices = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.env_indices = torch.arange(env.num_envs, device=env.device)

        # Cache gear assets
        self.gear_assets = {
            "gear_small": env.scene["factory_gear_small"],
            "gear_medium": env.scene["factory_gear_medium"],
            "gear_large": env.scene["factory_gear_large"],
        }

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Compute gear position in world frame.

        Args:
            env: Environment instance

        Returns:
            Gear position tensor of shape (num_envs, 3)
        """
        # Check if gear type manager exists
        # During initialization (shape checking), the manager may not exist yet
        if not hasattr(env, "_gear_type_manager"):
            # Return default shape during initialization
            return torch.zeros(env.num_envs, 3, device=env.device)

        gear_type_manager: randomize_gear_type = env._gear_type_manager
        # Get gear type indices directly as tensor (no Python loops!)
        self.gear_type_indices = gear_type_manager.get_all_gear_type_indices()

        # Stack all gear positions
        all_gear_positions = torch.stack(
            [
                self.gear_assets["gear_small"].data.root_pos_w.torch,
                self.gear_assets["gear_medium"].data.root_pos_w.torch,
                self.gear_assets["gear_large"].data.root_pos_w.torch,
            ],
            dim=1,
        )

        # Select gear positions using advanced indexing
        gear_positions = all_gear_positions[self.env_indices, self.gear_type_indices]

        return gear_positions - env.scene.env_origins


class gear_quat_w(ManagerTermBase):
    """Gear orientation in world frame.

    This class-based term returns the orientation of the active gear in each environment. It uses
    vectorized indexing to efficiently select the correct gear orientation based on the gear type
    (small, medium, or large) active in each environment. The quaternion is canonicalized to ensure
    the w component is positive, reducing observation variation for the policy.

    Returns:
        Gear orientation tensor as a quaternion (w, x, y, z) with shape (num_envs, 4).

    Raises:
        RuntimeError: If the gear type manager is not initialized in the environment.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        """Initialize the gear orientation observation term.

        Args:
            cfg: Observation term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Pre-allocate gear type mapping and indices
        self.gear_type_map = {"gear_small": 0, "gear_medium": 1, "gear_large": 2}
        self.gear_type_indices = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.env_indices = torch.arange(env.num_envs, device=env.device)

        # Cache gear assets
        self.gear_assets = {
            "gear_small": env.scene["factory_gear_small"],
            "gear_medium": env.scene["factory_gear_medium"],
            "gear_large": env.scene["factory_gear_large"],
        }

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Compute gear orientation in world frame.

        Args:
            env: Environment instance

        Returns:
            Gear orientation tensor of shape (num_envs, 4)
        """
        # Check if gear type manager exists
        # During initialization (shape checking), the manager may not exist yet
        if not hasattr(env, "_gear_type_manager"):
            # Return default shape during initialization (identity quaternion)
            default_quat = torch.zeros(env.num_envs, 4, device=env.device)
            default_quat[:, 3] = 1.0
            return default_quat

        gear_type_manager: randomize_gear_type = env._gear_type_manager
        # Get gear type indices directly as tensor (no Python loops!)
        self.gear_type_indices = gear_type_manager.get_all_gear_type_indices()

        # Stack all gear quaternions
        all_gear_quat = torch.stack(
            [
                self.gear_assets["gear_small"].data.root_quat_w.torch,
                self.gear_assets["gear_medium"].data.root_quat_w.torch,
                self.gear_assets["gear_large"].data.root_quat_w.torch,
            ],
            dim=1,
        )

        # Select gear quaternions using advanced indexing
        gear_quat = all_gear_quat[self.env_indices, self.gear_type_indices]

        # Ensure w component is positive (q and -q represent the same rotation)
        # Pick one canonical form to reduce observation variation seen by the policy
        w_negative = gear_quat[:, 3] < 0
        gear_positive_quat = gear_quat.clone()
        gear_positive_quat[w_negative] = -gear_quat[w_negative]

        return gear_positive_quat


class rigid_object_pos_w(ManagerTermBase):
    """Rigid object position in the environment frame, with optional local-frame offset.

    Generic observation term that returns the position of any
    :class:`~isaaclab.assets.RigidObject` in the environment frame. An optional
    3D offset can be applied in the object's local frame before subtracting the
    environment origin.

    Args:
        asset_cfg: The asset configuration. Required.
        offset: A 3D offset ``[x, y, z]`` [m] applied in the object's local frame.
            Defaults to ``[0, 0, 0]``.

    Returns:
        Object position tensor in the environment frame, shape ``[num_envs, 3]`` [m].
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        if "asset_cfg" not in cfg.params:
            raise ValueError("'asset_cfg' parameter is required in rigid_object_pos_w configuration.")
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject = env.scene[self.asset_cfg.name]

        offset = cfg.params.get("offset", [0.0, 0.0, 0.0])
        self.offset_tensor = torch.tensor(offset, device=env.device, dtype=torch.float32)
        self.leapp_input_name: str | None = cfg.params.get("leapp_input_name")

        self.identity_quat = (
            torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=env.device, dtype=torch.float32)
            .repeat(env.num_envs, 1)
            .contiguous()
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg | None = None,
        offset: list | None = None,
        leapp_input_name: str | None = None,
    ) -> torch.Tensor:
        real_env = _leapp_real_env(env)
        asset = real_env.scene[self.asset_cfg.name]
        obj_pos = _tensor_data_to_torch(asset.data.root_pos_w)
        obj_quat = _tensor_data_to_torch(asset.data.root_quat_w)

        if torch.any(self.offset_tensor != 0):
            offset_repeated = self.offset_tensor.unsqueeze(0).repeat(real_env.num_envs, 1)
            obj_pos, _ = combine_frame_transforms(obj_pos, obj_quat, offset_repeated, self.identity_quat)

        obj_pos = obj_pos - real_env.scene.env_origins
        if _is_leapp_export_env(env):
            from leapp import annotate
            from leapp.utils.tensor_description import TensorSemantics

            input_name = (
                leapp_input_name
                or self.leapp_input_name
                or f"{_deploy_object_input_base_name(self.asset_cfg.name)}_pos"
            )
            obj_pos = annotate.input_tensors(
                env.unwrapped.spec.id,
                TensorSemantics(
                    name=input_name,
                    ref=obj_pos,
                    kind=InputKindEnum.BODY_POSITION,
                    element_names=XYZ_ELEMENT_NAMES,
                    extra={"isaaclab_connection": f"observation:policy:{input_name}"},
                ),
            )
        return obj_pos


class rigid_object_quat_w(ManagerTermBase):
    """Rigid object orientation in the world frame.

    Generic observation term that returns the orientation of any
    :class:`~isaaclab.assets.RigidObject`. The quaternion is canonicalized so
    that the ``w`` component is positive, reducing observation variation seen
    by the policy.

    Args:
        asset_cfg: The asset configuration. Required.

    Returns:
        Object orientation as a quaternion ``(x, y, z, w)``, shape ``[num_envs, 4]``.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        if "asset_cfg" not in cfg.params:
            raise ValueError("'asset_cfg' parameter is required in rigid_object_quat_w configuration.")
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject = env.scene[self.asset_cfg.name]
        self.leapp_input_name: str | None = cfg.params.get("leapp_input_name")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg | None = None,
        leapp_input_name: str | None = None,
    ) -> torch.Tensor:
        real_env = _leapp_real_env(env)
        obj_quat = _tensor_data_to_torch(real_env.scene[self.asset_cfg.name].data.root_quat_w)
        if _is_leapp_export_env(env):
            from leapp import annotate
            from leapp.utils.tensor_description import TensorSemantics

            input_name = (
                leapp_input_name
                or self.leapp_input_name
                or f"{_deploy_object_input_base_name(self.asset_cfg.name)}_quat"
            )
            obj_quat = annotate.input_tensors(
                env.unwrapped.spec.id,
                TensorSemantics(
                    name=input_name,
                    ref=obj_quat,
                    kind=InputKindEnum.BODY_ROTATION,
                    element_names=QUAT_XYZW_ELEMENT_NAMES,
                    extra={"isaaclab_connection": f"observation:policy:{input_name}"},
                ),
            )

        sign = torch.where(obj_quat[:, 3:4] < 0, -1.0, 1.0)
        return obj_quat * sign


def _quat_to_rot_6d(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (x, y, z, w) to 6D rotation (Zhou et al.).

    Takes the first two rows of the 3x3 rotation matrix and flattens
    them into a 6-element vector per sample.

    Args:
        quat: Quaternion tensor of shape ``(..., 4)`` in ``(x, y, z, w)`` format.

    Returns:
        6D rotation tensor of shape ``(..., 6)``.
    """
    rot_mat = matrix_from_quat(quat)
    batch_shape = rot_mat.shape[:-2]
    return rot_mat[..., :2, :].clone().reshape(batch_shape + (6,))


class rigid_object_rot_6d_w(ManagerTermBase):
    """Rigid object 6D rotation in the world frame (Zhou et al.).

    Returns the first two rows of the 3x3 rotation matrix derived from
    the object's root quaternion, giving a continuous 6D rotation
    representation that avoids quaternion discontinuities.

    Args:
        asset_cfg: The asset configuration. Required.

    Returns:
        6D rotation tensor, shape ``[num_envs, 6]``.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if "asset_cfg" not in cfg.params:
            raise ValueError("'asset_cfg' parameter is required in rigid_object_rot_6d_w configuration.")
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject = env.scene[self.asset_cfg.name]
        self.leapp_input_name: str | None = cfg.params.get("leapp_input_name")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg | None = None,
        leapp_input_name: str | None = None,
    ) -> torch.Tensor:
        real_env = _leapp_real_env(env)
        obj_quat = _tensor_data_to_torch(real_env.scene[self.asset_cfg.name].data.root_quat_w)
        rot_6d = _quat_to_rot_6d(obj_quat)
        if _is_leapp_export_env(env):
            from leapp import annotate
            from leapp.utils.tensor_description import TensorSemantics

            input_name = (
                leapp_input_name
                or self.leapp_input_name
                or f"{_deploy_object_input_base_name(self.asset_cfg.name)}_rot_6d"
            )
            rot_6d = annotate.input_tensors(
                env.unwrapped.spec.id,
                TensorSemantics(
                    name=input_name,
                    ref=rot_6d,
                    kind=None,
                    element_names=ROT6D_ELEMENT_NAMES,
                    extra={"isaaclab_connection": f"observation:policy:{input_name}"},
                ),
            )
        return rot_6d


class eef_pos_w(ManagerTermBase):
    """End-effector position in the environment frame.

    Gets the position of a specified body on a robot articulation and
    returns it relative to the environment origin. An optional 3D offset can be
    applied in the body's local frame, e.g. to report the gripper tool-center
    point (TCP) rather than the raw flange.

    Args:
        asset_cfg: The robot articulation configuration. Required.
        body_name: Name of the end-effector body link. Required.
        offset: A 3D offset ``[x, y, z]`` [m] applied in the body's local frame.
            Defaults to ``[0, 0, 0]``.

    Returns:
        EEF position tensor, shape ``[num_envs, 3]`` [m].
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if "asset_cfg" not in cfg.params:
            raise ValueError("'asset_cfg' parameter is required in eef_pos_w configuration.")
        if "body_name" not in cfg.params:
            raise ValueError("'body_name' parameter is required in eef_pos_w configuration.")

        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.robot: Articulation = env.scene[self.asset_cfg.name]
        self.body_name: str = cfg.params["body_name"]
        self.body_idx = self.robot.find_bodies(self.body_name)[0][0]

        offset = cfg.params.get("offset", [0.0, 0.0, 0.0])
        self.offset_tensor = torch.tensor(offset, device=env.device, dtype=torch.float32)
        self.identity_quat = (
            torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=env.device, dtype=torch.float32)
            .repeat(env.num_envs, 1)
            .contiguous()
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg | None = None,
        body_name: str | None = None,
        offset: list | None = None,
    ) -> torch.Tensor:
        real_env = _leapp_real_env(env)
        robot = real_env.scene[self.asset_cfg.name]
        body_pos = _tensor_data_to_torch(robot.data.body_pos_w)[:, self.body_idx, :]

        if torch.any(self.offset_tensor != 0):
            body_quat = _tensor_data_to_torch(robot.data.body_quat_w)[:, self.body_idx, :]
            offset_repeated = self.offset_tensor.unsqueeze(0).repeat(real_env.num_envs, 1)
            identity_quat = self.identity_quat[: real_env.num_envs]
            body_pos, _ = combine_frame_transforms(body_pos, body_quat, offset_repeated, identity_quat)

        body_pos = body_pos - real_env.scene.env_origins
        if _is_leapp_export_env(env):
            from leapp import annotate
            from leapp.utils.tensor_description import TensorSemantics

            body_pos = annotate.input_tensors(
                env.unwrapped.spec.id,
                TensorSemantics(
                    name="eef_pos",
                    ref=body_pos,
                    kind=InputKindEnum.BODY_POSITION,
                    element_names=XYZ_ELEMENT_NAMES,
                    extra={"isaaclab_connection": "observation:policy:eef_pos"},
                ),
            )
        return body_pos


class eef_rot_6d_w(ManagerTermBase):
    """End-effector 6D rotation in the world frame (Zhou et al.).

    Gets the quaternion of a specified body on a robot articulation and
    converts it to a continuous 6D rotation representation.

    Args:
        asset_cfg: The robot articulation configuration. Required.
        body_name: Name of the end-effector body link. Required.

    Returns:
        6D rotation tensor, shape ``[num_envs, 6]``.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if "asset_cfg" not in cfg.params:
            raise ValueError("'asset_cfg' parameter is required in eef_rot_6d_w configuration.")
        if "body_name" not in cfg.params:
            raise ValueError("'body_name' parameter is required in eef_rot_6d_w configuration.")

        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.robot: Articulation = env.scene[self.asset_cfg.name]
        self.body_name: str = cfg.params["body_name"]
        self.body_idx = self.robot.find_bodies(self.body_name)[0][0]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg | None = None,
        body_name: str | None = None,
    ) -> torch.Tensor:
        real_env = _leapp_real_env(env)
        robot = real_env.scene[self.asset_cfg.name]
        body_quat = _tensor_data_to_torch(robot.data.body_quat_w)[:, self.body_idx, :]
        rot_6d = _quat_to_rot_6d(body_quat)
        if _is_leapp_export_env(env):
            from leapp import annotate
            from leapp.utils.tensor_description import TensorSemantics

            rot_6d = annotate.input_tensors(
                env.unwrapped.spec.id,
                TensorSemantics(
                    name="eef_rot_6d",
                    ref=rot_6d,
                    kind=None,
                    element_names=ROT6D_ELEMENT_NAMES,
                    extra={"isaaclab_connection": "observation:policy:eef_rot_6d"},
                ),
            )
        return rot_6d
