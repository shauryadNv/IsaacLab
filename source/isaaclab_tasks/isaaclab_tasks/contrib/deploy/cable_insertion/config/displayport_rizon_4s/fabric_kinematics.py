# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-independent serial-chain kinematics for DisplayPort controller experiments."""

from __future__ import annotations

from typing import NamedTuple

import torch

from isaaclab.utils import math as math_utils


class SerialChainKinematicsResult(NamedTuple):
    r"""Kinematic quantities evaluated at one batched joint state.

    Attributes:
        tcp_pose: TCP pose ``(position, quaternion_xyzw)`` in the base frame
            [m, unitless], shape ``(..., 7)``.
        jacobian: TCP geometric Jacobian in the base frame, ordered as linear
            then angular velocity [m/rad, unitless], shape ``(..., 6, 7)``.
        jacobian_dot_times_velocity: Product :math:`\dot{J}(q, \dot{q})\dot{q}`
            in the base frame [m/s^2, rad/s^2], shape ``(..., 6)``.
    """

    tcp_pose: torch.Tensor
    jacobian: torch.Tensor
    jacobian_dot_times_velocity: torch.Tensor


class BatchedSerialChainKinematics(torch.nn.Module):
    r"""Evaluate fixed-base, seven-revolute-joint serial-chain kinematics.

    The implementation is independent of the simulation backend and stores only
    rigid transforms and joint axes. Transform ``T_A_B`` maps coordinates from
    frame ``B`` into frame ``A``. For joint ``i``, the child-link transform is

    .. math::

        T_{P_i C_i}(q_i) = T_{P_i J_i}\,R(a_i, q_i)\,T_{C_i J_i}^{-1},

    where ``parent_to_joint[i]`` is :math:`T_{P_i J_i}`,
    ``child_to_joint[i]`` is :math:`T_{C_i J_i}`, and ``joint_axes[i]`` is
    expressed in :math:`J_i`. ``child_to_tcp`` is :math:`T_{C_7 T}`.

    The Jacobian is a spatial geometric Jacobian at the TCP, expressed in the
    base frame and ordered ``(linear, angular)``. The
    :math:`\dot{J}\dot{q}` result is analytic for a fixed-base revolute chain;
    it does not use finite differences or simulator state.
    """

    _NUM_JOINTS = 7

    def __init__(
        self,
        parent_to_joint: torch.Tensor,
        child_to_joint: torch.Tensor,
        joint_axes: torch.Tensor,
        child_to_tcp: torch.Tensor,
    ) -> None:
        """Initialize the serial-chain description.

        Args:
            parent_to_joint: Parent-link-to-joint rigid transforms
                :math:`T_{P_i J_i}` [m, unitless], shape ``(7, 4, 4)``.
            child_to_joint: Child-link-to-joint rigid transforms
                :math:`T_{C_i J_i}` [m, unitless], shape ``(7, 4, 4)``.
            joint_axes: Revolute axes expressed in each joint frame [unitless],
                shape ``(7, 3)``. Axes are normalized during initialization.
            child_to_tcp: Final-child-link-to-TCP rigid transform
                :math:`T_{C_7 T}` [m, unitless], shape ``(4, 4)``.
        """
        super().__init__()

        parent_to_joint = self._as_floating_tensor(parent_to_joint)
        device, dtype = parent_to_joint.device, parent_to_joint.dtype
        child_to_joint = self._as_floating_tensor(child_to_joint).to(device=device, dtype=dtype)
        joint_axes = self._as_floating_tensor(joint_axes).to(device=device, dtype=dtype)
        child_to_tcp = self._as_floating_tensor(child_to_tcp).to(device=device, dtype=dtype)

        expected_transform_shape = (self._NUM_JOINTS, 4, 4)
        if parent_to_joint.shape != expected_transform_shape:
            raise ValueError(
                f"Expected parent_to_joint shape {expected_transform_shape}, received {tuple(parent_to_joint.shape)}."
            )
        if child_to_joint.shape != expected_transform_shape:
            raise ValueError(
                f"Expected child_to_joint shape {expected_transform_shape}, received {tuple(child_to_joint.shape)}."
            )
        if joint_axes.shape != (self._NUM_JOINTS, 3):
            raise ValueError(f"Expected joint_axes shape {(self._NUM_JOINTS, 3)}, received {tuple(joint_axes.shape)}.")
        if child_to_tcp.shape != (4, 4):
            raise ValueError(f"Expected child_to_tcp shape {(4, 4)}, received {tuple(child_to_tcp.shape)}.")

        self._validate_rigid_transforms("parent_to_joint", parent_to_joint)
        self._validate_rigid_transforms("child_to_joint", child_to_joint)
        self._validate_rigid_transforms("child_to_tcp", child_to_tcp.unsqueeze(0))

        axis_norm = torch.linalg.vector_norm(joint_axes, dim=-1, keepdim=True)
        if torch.any(axis_norm <= 1.0e-8):
            raise ValueError("Every revolute joint axis must have a non-zero norm.")
        joint_axes = joint_axes / axis_norm

        self.register_buffer("parent_to_joint", parent_to_joint.clone())
        self.register_buffer("joint_to_child", self._invert_rigid_transform(child_to_joint))
        self.register_buffer("joint_axes", joint_axes.clone())
        self.register_buffer("child_to_tcp", child_to_tcp.clone())

    @property
    def num_joints(self) -> int:
        """Number of revolute joints in the chain."""
        return self._NUM_JOINTS

    def forward(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> SerialChainKinematicsResult:
        r"""Evaluate TCP pose, geometric Jacobian, and :math:`\dot{J}\dot{q}`.

        Args:
            joint_pos: Revolute joint positions [rad], shape ``(..., 7)``.
            joint_vel: Revolute joint velocities [rad/s], shape ``(..., 7)``.

        Returns:
            Batched kinematic quantities with the same leading dimensions as
            ``joint_pos``.
        """
        if joint_pos.shape != joint_vel.shape:
            raise ValueError(
                "joint_pos and joint_vel must have identical shapes, received "
                f"{tuple(joint_pos.shape)} and {tuple(joint_vel.shape)}."
            )
        if joint_pos.ndim < 1 or joint_pos.shape[-1] != self._NUM_JOINTS:
            raise ValueError(
                f"Expected joint state shape (..., {self._NUM_JOINTS}), received {tuple(joint_pos.shape)}."
            )
        if joint_pos.device != self.parent_to_joint.device or joint_pos.dtype != self.parent_to_joint.dtype:
            raise ValueError(
                "Joint state device and dtype must match the kinematics module; call module.to(device, dtype) first."
            )

        batch_shape = joint_pos.shape[:-1]
        joint_pos_flat = joint_pos.reshape(-1, self._NUM_JOINTS)
        joint_vel_flat = joint_vel.reshape(-1, self._NUM_JOINTS)
        batch_size = joint_pos_flat.shape[0]

        base_to_parent = torch.eye(4, dtype=joint_pos.dtype, device=joint_pos.device).expand(batch_size, 4, 4)
        joint_origins = []
        joint_axes = []

        for joint_index in range(self._NUM_JOINTS):
            base_to_joint = base_to_parent @ self.parent_to_joint[joint_index]
            joint_origins.append(base_to_joint[:, :3, 3])
            joint_axes.append((base_to_joint[:, :3, :3] @ self.joint_axes[joint_index].reshape(1, 3, 1)).squeeze(-1))
            joint_rotation = self._axis_angle_transform(self.joint_axes[joint_index], joint_pos_flat[:, joint_index])
            base_to_parent = base_to_joint @ joint_rotation @ self.joint_to_child[joint_index]

        base_to_tcp = base_to_parent @ self.child_to_tcp
        tcp_position = base_to_tcp[:, :3, 3]
        tcp_quaternion = math_utils.quat_unique(math_utils.quat_from_matrix(base_to_tcp[:, :3, :3]))

        joint_origins_tensor = torch.stack(joint_origins, dim=1)
        joint_axes_tensor = torch.stack(joint_axes, dim=1)
        lever_arm = tcp_position.unsqueeze(1) - joint_origins_tensor
        linear_jacobian_columns = torch.linalg.cross(joint_axes_tensor, lever_arm, dim=-1)
        jacobian = torch.cat((linear_jacobian_columns.transpose(1, 2), joint_axes_tensor.transpose(1, 2)), dim=1)

        jacobian_dot_times_velocity = self._compute_jacobian_dot_times_velocity(
            joint_origins_tensor,
            joint_axes_tensor,
            tcp_position,
            linear_jacobian_columns,
            joint_vel_flat,
        )

        return SerialChainKinematicsResult(
            tcp_pose=torch.cat((tcp_position, tcp_quaternion), dim=-1).reshape(batch_shape + (7,)),
            jacobian=jacobian.reshape(batch_shape + (6, self._NUM_JOINTS)),
            jacobian_dot_times_velocity=jacobian_dot_times_velocity.reshape(batch_shape + (6,)),
        )

    def _compute_jacobian_dot_times_velocity(
        self,
        joint_origins: torch.Tensor,
        joint_axes: torch.Tensor,
        tcp_position: torch.Tensor,
        linear_jacobian_columns: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute analytic :math:`\dot{J}\dot{q}` for the fixed-base chain."""
        angular_velocity_contributions = joint_axes * joint_vel.unsqueeze(-1)
        parent_angular_velocity = torch.cumsum(angular_velocity_contributions, dim=1)
        parent_angular_velocity = parent_angular_velocity - angular_velocity_contributions
        joint_axis_derivative = torch.linalg.cross(parent_angular_velocity, joint_axes, dim=-1)

        joint_origin_velocity = torch.zeros_like(joint_origins)
        for joint_index in range(1, self._NUM_JOINTS):
            upstream_lever_arm = joint_origins[:, joint_index : joint_index + 1, :] - joint_origins[:, :joint_index, :]
            upstream_linear_columns = torch.linalg.cross(joint_axes[:, :joint_index, :], upstream_lever_arm, dim=-1)
            joint_origin_velocity[:, joint_index, :] = torch.sum(
                upstream_linear_columns * joint_vel[:, :joint_index].unsqueeze(-1), dim=1
            )

        tcp_linear_velocity = torch.sum(linear_jacobian_columns * joint_vel.unsqueeze(-1), dim=1)
        lever_arm = tcp_position.unsqueeze(1) - joint_origins
        linear_jacobian_derivative = torch.linalg.cross(joint_axis_derivative, lever_arm, dim=-1)
        linear_jacobian_derivative += torch.linalg.cross(
            joint_axes,
            tcp_linear_velocity.unsqueeze(1) - joint_origin_velocity,
            dim=-1,
        )

        linear_acceleration = torch.sum(linear_jacobian_derivative * joint_vel.unsqueeze(-1), dim=1)
        angular_acceleration = torch.sum(joint_axis_derivative * joint_vel.unsqueeze(-1), dim=1)
        return torch.cat((linear_acceleration, angular_acceleration), dim=-1)

    def _axis_angle_transform(self, axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """Construct batched rigid transforms for rotation about a fixed axis."""
        skew_axis = self._skew(axis)
        skew_axis_squared = skew_axis @ skew_axis
        sine = torch.sin(angle).reshape(-1, 1, 1)
        cosine = torch.cos(angle).reshape(-1, 1, 1)
        identity_rotation = torch.eye(3, dtype=angle.dtype, device=angle.device).reshape(1, 3, 3)
        rotation = identity_rotation + sine * skew_axis + (1.0 - cosine) * skew_axis_squared

        transform = torch.eye(4, dtype=angle.dtype, device=angle.device).expand(angle.shape[0], 4, 4).clone()
        transform[:, :3, :3] = rotation
        return transform

    @staticmethod
    def _skew(vector: torch.Tensor) -> torch.Tensor:
        """Return the skew-symmetric cross-product matrix of one 3-vector."""
        zero = torch.zeros((), dtype=vector.dtype, device=vector.device)
        x, y, z = vector.unbind()
        return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)

    @staticmethod
    def _invert_rigid_transform(transform: torch.Tensor) -> torch.Tensor:
        """Invert batched homogeneous rigid transforms."""
        rotation_transpose = transform[..., :3, :3].transpose(-1, -2)
        translation = transform[..., :3, 3]
        inverse = torch.zeros_like(transform)
        inverse[..., :3, :3] = rotation_transpose
        inverse[..., :3, 3] = -(rotation_transpose @ translation.unsqueeze(-1)).squeeze(-1)
        inverse[..., 3, 3] = 1.0
        return inverse

    @staticmethod
    def _as_floating_tensor(value: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(value)
        if not tensor.is_floating_point():
            tensor = tensor.to(dtype=torch.float32)
        return tensor

    @staticmethod
    def _validate_rigid_transforms(name: str, transform: torch.Tensor) -> None:
        """Validate homogeneous rigid transforms at module construction time."""
        expected_bottom_row = torch.zeros_like(transform[..., 3, :])
        expected_bottom_row[..., 3] = 1.0
        if not torch.allclose(transform[..., 3, :], expected_bottom_row, atol=1.0e-6, rtol=0.0):
            raise ValueError(f"{name} must contain homogeneous rigid transforms with bottom row [0, 0, 0, 1].")

        rotation = transform[..., :3, :3]
        identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device).expand_as(rotation)
        if not torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=1.0e-5, rtol=1.0e-5):
            raise ValueError(f"{name} contains a non-orthonormal rotation matrix.")
        if not torch.all(torch.linalg.det(rotation) > 0.0):
            raise ValueError(f"{name} contains a reflection rather than a proper rotation.")
