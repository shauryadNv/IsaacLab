# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Property tests for the task-private DisplayPort serial-chain kinematics."""

import pytest
import torch

from isaaclab.utils import math as math_utils

from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.fabric_kinematics import (
    BatchedSerialChainKinematics,
)


def _skew(vector: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros((), dtype=vector.dtype, device=vector.device)
    x, y, z = vector.unbind()
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)


def _transform(
    translation: tuple[float, float, float],
    axis: tuple[float, float, float],
    angle: float,
    *,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    axis_tensor = torch.tensor(axis, dtype=dtype, device=device)
    axis_tensor = axis_tensor / torch.linalg.vector_norm(axis_tensor)
    skew_axis = _skew(axis_tensor)
    rotation = (
        torch.eye(3, dtype=dtype, device=device)
        + torch.sin(torch.tensor(angle, dtype=dtype, device=device)) * skew_axis
        + (1.0 - torch.cos(torch.tensor(angle, dtype=dtype, device=device))) * (skew_axis @ skew_axis)
    )
    transform = torch.eye(4, dtype=dtype, device=device)
    transform[:3, :3] = rotation
    transform[:3, 3] = torch.tensor(translation, dtype=dtype, device=device)
    return transform


def _make_kinematics(*, dtype: torch.dtype = torch.float64, device: str = "cpu") -> BatchedSerialChainKinematics:
    parent_translations = (
        (0.00, 0.00, 0.18),
        (0.04, 0.00, 0.16),
        (-0.03, 0.02, 0.15),
        (0.02, -0.03, 0.13),
        (0.01, 0.02, 0.11),
        (-0.02, 0.01, 0.09),
        (0.01, -0.01, 0.07),
    )
    fixed_rotation_axes = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, -1.0, 0.5),
    )
    fixed_rotation_angles = (0.08, -0.06, 0.04, 0.05, -0.07, 0.03, -0.02)
    parent_to_joint = torch.stack(
        [
            _transform(translation, axis, angle, dtype=dtype, device=device)
            for translation, axis, angle in zip(
                parent_translations, fixed_rotation_axes, fixed_rotation_angles, strict=True
            )
        ]
    )

    child_to_joint = torch.stack(
        [
            _transform(
                (0.004 * ((index % 3) - 1), -0.003 * (index % 2), 0.002 * ((index + 1) % 2)),
                fixed_rotation_axes[(index + 2) % 7],
                0.015 * ((index % 3) - 1),
                dtype=dtype,
                device=device,
            )
            for index in range(7)
        ]
    )
    joint_axes = torch.tensor(
        (
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (0.2, -0.3, 1.0),
        ),
        dtype=dtype,
        device=device,
    )
    child_to_tcp = _transform((0.035, -0.018, 0.145), (1.0, 1.0, 0.0), 0.09, dtype=dtype, device=device)
    return BatchedSerialChainKinematics(parent_to_joint, child_to_joint, joint_axes, child_to_tcp)


def _sample_joint_state(
    batch_size: int, *, dtype: torch.dtype = torch.float64, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(19)
    joint_pos = (torch.rand(batch_size, 7, generator=generator, dtype=dtype, device=device) - 0.5) * 1.4
    joint_vel = (torch.rand(batch_size, 7, generator=generator, dtype=dtype, device=device) - 0.5) * 1.2
    return joint_pos, joint_vel


def _rotation_from_pose(pose: torch.Tensor) -> torch.Tensor:
    return math_utils.matrix_from_quat(pose[..., 3:7])


def _vee(skew_matrix: torch.Tensor) -> torch.Tensor:
    return torch.stack((skew_matrix[..., 2, 1], skew_matrix[..., 0, 2], skew_matrix[..., 1, 0]), dim=-1)


def test_batch_and_singleton_evaluations_match():
    kinematics = _make_kinematics()
    joint_pos, joint_vel = _sample_joint_state(5)

    batched = kinematics(joint_pos, joint_vel)
    for index in range(joint_pos.shape[0]):
        singleton = kinematics(joint_pos[index], joint_vel[index])
        torch.testing.assert_close(singleton.tcp_pose, batched.tcp_pose[index])
        torch.testing.assert_close(singleton.jacobian, batched.jacobian[index])
        torch.testing.assert_close(
            singleton.jacobian_dot_times_velocity,
            batched.jacobian_dot_times_velocity[index],
        )


def test_geometric_jacobian_matches_pose_finite_differences():
    kinematics = _make_kinematics()
    joint_pos, _ = _sample_joint_state(3)
    zero_velocity = torch.zeros_like(joint_pos)
    nominal = kinematics(joint_pos, zero_velocity)
    nominal_rotation = _rotation_from_pose(nominal.tcp_pose)
    epsilon = 1.0e-6

    for joint_index in range(7):
        perturbation = torch.zeros_like(joint_pos)
        perturbation[:, joint_index] = epsilon
        plus = kinematics(joint_pos + perturbation, zero_velocity)
        minus = kinematics(joint_pos - perturbation, zero_velocity)

        linear_derivative = (plus.tcp_pose[:, :3] - minus.tcp_pose[:, :3]) / (2.0 * epsilon)
        rotation_derivative = (_rotation_from_pose(plus.tcp_pose) - _rotation_from_pose(minus.tcp_pose)) / (
            2.0 * epsilon
        )
        angular_derivative = _vee(rotation_derivative @ nominal_rotation.transpose(-1, -2))

        torch.testing.assert_close(nominal.jacobian[:, :3, joint_index], linear_derivative, rtol=2.0e-7, atol=2.0e-8)
        torch.testing.assert_close(nominal.jacobian[:, 3:, joint_index], angular_derivative, rtol=2.0e-7, atol=2.0e-8)


def test_jacobian_dot_times_velocity_matches_directional_finite_difference():
    kinematics = _make_kinematics()
    joint_pos, joint_vel = _sample_joint_state(4)
    epsilon = 1.0e-6

    plus = kinematics(joint_pos + epsilon * joint_vel, joint_vel)
    minus = kinematics(joint_pos - epsilon * joint_vel, joint_vel)
    jacobian_derivative = (plus.jacobian - minus.jacobian) / (2.0 * epsilon)
    finite_difference = (jacobian_derivative @ joint_vel.unsqueeze(-1)).squeeze(-1)
    analytic = kinematics(joint_pos, joint_vel).jacobian_dot_times_velocity

    torch.testing.assert_close(analytic, finite_difference, rtol=5.0e-7, atol=5.0e-8)


def test_jacobian_pullback_preserves_virtual_work():
    kinematics = _make_kinematics()
    joint_pos, joint_vel = _sample_joint_state(6)
    result = kinematics(joint_pos, joint_vel)
    generator = torch.Generator().manual_seed(23)
    wrench = torch.randn(6, 6, generator=generator, dtype=joint_pos.dtype)

    generalized_force = (result.jacobian.transpose(-1, -2) @ wrench.unsqueeze(-1)).squeeze(-1)
    tcp_twist = (result.jacobian @ joint_vel.unsqueeze(-1)).squeeze(-1)
    joint_power = torch.sum(joint_vel * generalized_force, dim=-1)
    tcp_power = torch.sum(tcp_twist * wrench, dim=-1)

    torch.testing.assert_close(joint_power, tcp_power, rtol=1.0e-12, atol=1.0e-12)


def test_invalid_joint_axis_is_rejected():
    kinematics = _make_kinematics()
    invalid_axes = kinematics.joint_axes.clone()
    invalid_axes[3] = 0.0

    with pytest.raises(ValueError, match="non-zero norm"):
        BatchedSerialChainKinematics(
            kinematics.parent_to_joint,
            torch.linalg.inv(kinematics.joint_to_child),
            invalid_axes,
            kinematics.child_to_tcp,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_and_cuda_traces_match():
    cpu_kinematics = _make_kinematics(dtype=torch.float64, device="cpu")
    cuda_kinematics = _make_kinematics(dtype=torch.float64, device="cuda:0")
    joint_pos, joint_vel = _sample_joint_state(32, dtype=torch.float64, device="cpu")

    cpu_result = cpu_kinematics(joint_pos, joint_vel)
    cuda_result = cuda_kinematics(joint_pos.cuda(), joint_vel.cuda())

    torch.testing.assert_close(cpu_result.tcp_pose, cuda_result.tcp_pose.cpu(), rtol=1.0e-11, atol=1.0e-12)
    torch.testing.assert_close(cpu_result.jacobian, cuda_result.jacobian.cpu(), rtol=1.0e-11, atol=1.0e-12)
    torch.testing.assert_close(
        cpu_result.jacobian_dot_times_velocity,
        cuda_result.jacobian_dot_times_velocity.cpu(),
        rtol=1.0e-11,
        atol=1.0e-12,
    )
