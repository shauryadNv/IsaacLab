# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Property tests for the task-private DisplayPort joint fabric controller."""

from dataclasses import fields

import pytest
import torch

from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.joint_fabric_controller import (
    JointFabricController,
    JointFabricControllerCfg,
    JointFabricTaskKinematics,
)

_NUM_JOINTS = 7


def _make_cfg(**overrides) -> JointFabricControllerCfg:
    cfg = JointFabricControllerCfg()
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return cfg


def _make_controller(
    *,
    num_envs: int = 1,
    device: str = "cpu",
    cfg: JointFabricControllerCfg | None = None,
    lower: float = -2.0,
    upper: float = 2.0,
) -> JointFabricController:
    return JointFabricController(
        cfg if cfg is not None else JointFabricControllerCfg(),
        num_envs,
        device,
        (lower,) * _NUM_JOINTS,
        (upper,) * _NUM_JOINTS,
    )


def _zero_kinematics(controller: JointFabricController) -> JointFabricTaskKinematics:
    kwargs = {"dtype": controller.joint_position.dtype, "device": controller.device}
    return JointFabricTaskKinematics(
        transverse_error=torch.zeros(controller.num_envs, 2, **kwargs),
        transverse_jacobian=torch.zeros(controller.num_envs, 2, _NUM_JOINTS, **kwargs),
        transverse_jacobian_dot_velocity=torch.zeros(controller.num_envs, 2, **kwargs),
        tcp_jacobian=torch.zeros(controller.num_envs, 6, _NUM_JOINTS, **kwargs),
    )


def _reset_zero(controller: JointFabricController) -> None:
    controller.reset(torch.zeros_like(controller.joint_position))


def _unconstrained_cfg(**overrides) -> JointFabricControllerCfg:
    values = {
        "root_metric": (1.0,) * _NUM_JOINTS,
        "transverse_funnel_gain": 0.0,
        "transverse_damping": (0.0, 0.0),
        "posture_stiffness": (0.0,) * _NUM_JOINTS,
        "root_damping": (0.0,) * _NUM_JOINTS,
        "geometric_damping": 0.0,
        "joint_limit_barrier_metric": 0.0,
        "joint_limit_barrier_stiffness": 0.0,
        "joint_limit_barrier_damping": 0.0,
        "max_joint_velocity": (100.0,) * _NUM_JOINTS,
        "max_joint_acceleration": (100.0,) * _NUM_JOINTS,
        "max_joint_jerk": (1000.0,) * _NUM_JOINTS,
        "solve_regularization": 0.0,
    }
    values.update(overrides)
    return _make_cfg(**values)


def test_hd2_homogeneity_natural_form_pullback_and_path_consistency():
    controller = _make_controller(cfg=_make_cfg(transverse_metric=(2.0, 3.0)))
    error = torch.tensor([[0.012, -0.007]], dtype=torch.float32)
    jacobian = torch.zeros(1, 2, _NUM_JOINTS)
    jacobian[0, 0, 0] = 1.2
    jacobian[0, 1, 1] = -0.7
    jacobian_dot_velocity = torch.tensor([[0.03, -0.02]])
    joint_velocity = torch.tensor([[0.4, -0.3, 0.1, 0.0, 0.0, 0.0, 0.0]])

    metric, natural_force, funnel = controller.compute_transverse_natural_form(
        error, jacobian, jacobian_dot_velocity, joint_velocity
    )
    _, _, scaled_funnel = controller.compute_transverse_natural_form(
        error, jacobian, jacobian_dot_velocity, 2.5 * joint_velocity
    )

    task_metric = torch.diag(torch.tensor((2.0, 3.0))).unsqueeze(0)
    expected_metric = jacobian.transpose(1, 2) @ task_metric @ jacobian
    expected_force = (
        jacobian.transpose(1, 2) @ (task_metric @ (jacobian_dot_velocity - funnel).unsqueeze(-1))
    ).squeeze(-1)
    torch.testing.assert_close(metric, expected_metric)
    torch.testing.assert_close(natural_force, expected_force)
    torch.testing.assert_close(scaled_funnel, 2.5**2 * funnel, rtol=2.0e-6, atol=1.0e-7)
    assert torch.sum(error * funnel).item() < 0.0

    path_acceleration = torch.zeros_like(joint_velocity)
    path_acceleration[:, :2] = torch.linalg.solve(metric[:, :2, :2], -natural_force[:, :2])
    task_acceleration = (jacobian @ path_acceleration.unsqueeze(-1)).squeeze(-1) + jacobian_dot_velocity
    torch.testing.assert_close(task_acceleration, funnel, rtol=2.0e-5, atol=1.0e-6)


def test_wrench_pullback_is_bounded_and_preserves_virtual_work():
    cfg = _make_cfg(wrench_scale=(0.5, 0.4, 0.3, 0.05, 0.04, 0.03))
    controller = _make_controller(num_envs=4, cfg=cfg)
    generator = torch.Generator().manual_seed(17)
    action = 3.0 * torch.randn(4, 6, generator=generator)
    jacobian = torch.randn(4, 6, _NUM_JOINTS, generator=generator)
    joint_velocity = torch.randn(4, _NUM_JOINTS, generator=generator)

    wrench, generalized_force = controller.compute_bounded_wrench(action, jacobian)
    scale = torch.tensor(cfg.wrench_scale).reshape(1, -1)
    torch.testing.assert_close(wrench, torch.clamp(action, -1.0, 1.0) * scale)
    assert torch.all(torch.abs(wrench) <= scale + 1.0e-7)

    tcp_twist = (jacobian @ joint_velocity.unsqueeze(-1)).squeeze(-1)
    joint_power = torch.sum(joint_velocity * generalized_force, dim=-1)
    task_power = torch.sum(tcp_twist * wrench, dim=-1)
    torch.testing.assert_close(joint_power, task_power, rtol=2.0e-5, atol=2.0e-5)


def test_energization_is_orthogonal_in_configured_kinetic_metric():
    cfg = _unconstrained_cfg(
        kinetic_energy_weights=(1.0, 2.0, 0.5, 1.5, 0.8, 1.2, 0.7),
        root_metric=(0.2,) * _NUM_JOINTS,
        transverse_funnel_gain=0.7,
    )
    controller = _make_controller(cfg=cfg)
    position = torch.zeros(1, _NUM_JOINTS)
    velocity = torch.tensor([[0.18, -0.12, 0.08, 0.03, -0.02, 0.04, -0.05]])
    controller.reset(position, velocity)
    kinematics = _zero_kinematics(controller)
    kinematics.transverse_error[:] = torch.tensor([[0.01, -0.008]])
    kinematics.transverse_jacobian[0, 0, 0] = 1.0
    kinematics.transverse_jacobian[0, 1, 1] = 1.0

    controller.step(torch.zeros(1, 6), kinematics)

    assert abs(controller.telemetry.energy_residual.item()) < 2.0e-6
    assert controller.telemetry.metric_min_cholesky_diagonal.item() > 0.0
    assert not controller.faulted.item()


def test_joint_limit_gradient_and_damping_point_in_safe_directions():
    controller = _make_controller(num_envs=3)
    center = torch.zeros(_NUM_JOINTS)
    lower = controller.safe_joint_lower_limits[1] + 0.01
    upper = controller.safe_joint_upper_limits[2] - 0.01
    position = torch.stack((center, lower, upper))
    velocity = torch.stack((center, torch.full_like(center, -0.2), torch.full_like(center, 0.2)))

    metric, gradient, damping = controller.compute_joint_limit_terms(position, velocity)

    assert torch.all(metric >= 0.0)
    assert torch.all(metric[1] > metric[0])
    assert torch.all(metric[2] > metric[0])
    assert torch.all(gradient[1] < 0.0)
    assert torch.all(gradient[2] > 0.0)
    torch.testing.assert_close(gradient[0], torch.zeros_like(gradient[0]), atol=1.0e-7, rtol=0.0)
    assert torch.all(damping[1] < 0.0)
    assert torch.all(damping[2] > 0.0)
    # The dynamics subtract both terms, so these signs push away from the
    # corresponding limit and oppose outward velocity.


def test_exact_constant_jerk_endpoint_update_and_raw_saturation_count():
    cfg = _unconstrained_cfg(dt=0.1, wrench_scale=(1.0,) * 6)
    controller = _make_controller(cfg=cfg)
    _reset_zero(controller)
    kinematics = _zero_kinematics(controller)
    kinematics.tcp_jacobian[0, 0, 0] = 1.0
    action = torch.tensor([[0.25, 0.0, 0.0, 0.0, 0.0, 0.0]])

    previous_acceleration = torch.zeros(1, _NUM_JOINTS)
    previous_acceleration[0, 0] = 0.1
    controller._joint_acceleration[:] = previous_acceleration

    controller.step(action, kinematics)

    expected_acceleration = torch.zeros(1, _NUM_JOINTS)
    expected_acceleration[0, 0] = 0.25
    expected_velocity = 0.5 * 0.1 * (previous_acceleration + expected_acceleration)
    expected_position = 0.1**2 * (previous_acceleration / 3.0 + expected_acceleration / 6.0)
    torch.testing.assert_close(controller.joint_acceleration, expected_acceleration)
    torch.testing.assert_close(controller.joint_velocity, expected_velocity)
    torch.testing.assert_close(controller.joint_position, expected_position)
    assert controller.telemetry.action_saturation_count.item() == 0

    controller.reset(torch.zeros(1, _NUM_JOINTS))
    saturated_action = torch.tensor([[1.001, -1.0, 1.0, 0.0, 0.0, 0.0]])
    controller.step(saturated_action, kinematics)
    assert controller.telemetry.action_saturation_count.item() == 1


def test_acceleration_projection_intersects_acceleration_jerk_velocity_and_position_bounds():
    def project(
        cfg: JointFabricControllerCfg,
        *,
        position: float = 0.0,
        velocity: float = 0.0,
        previous_acceleration: float = 0.0,
        nominal_acceleration: float = 10.0,
        expected_infeasible: bool = False,
    ) -> tuple[float, tuple[int, int, int, int]]:
        controller = _make_controller(cfg=cfg, lower=-1.0, upper=1.0)
        q = torch.zeros(1, _NUM_JOINTS)
        v = torch.zeros_like(q)
        q[0, 0] = position
        v[0, 0] = velocity
        controller.reset(q, v)
        controller._joint_acceleration[0, 0] = previous_acceleration
        nominal = torch.zeros_like(q)
        nominal[0, 0] = nominal_acceleration
        projected, infeasible, accel, jerk, velocity_count, position_count = controller._project_acceleration(nominal)
        assert bool(infeasible.item()) is expected_infeasible
        return projected[0, 0].item(), (
            accel.item(),
            jerk.item(),
            velocity_count.item(),
            position_count.item(),
        )

    acceleration_cfg = _unconstrained_cfg(
        dt=0.1,
        max_joint_acceleration=(1.0,) * _NUM_JOINTS,
        max_joint_jerk=(1000.0,) * _NUM_JOINTS,
    )
    value, counts = project(acceleration_cfg)
    assert value == pytest.approx(1.0)
    assert counts[0] == 1

    jerk_cfg = _unconstrained_cfg(
        dt=0.1,
        max_joint_acceleration=(100.0,) * _NUM_JOINTS,
        max_joint_jerk=(2.0,) * _NUM_JOINTS,
    )
    value, counts = project(jerk_cfg)
    assert value == pytest.approx(0.2)
    assert counts[1] == 1

    velocity_cfg = _unconstrained_cfg(
        dt=0.1,
        max_joint_velocity=(0.5,) * _NUM_JOINTS,
        max_joint_acceleration=(100.0,) * _NUM_JOINTS,
        max_joint_jerk=(1000.0,) * _NUM_JOINTS,
    )
    value, counts = project(velocity_cfg, velocity=0.49)
    assert value == pytest.approx(0.2, abs=1.0e-5)
    assert counts[2] == 1

    value, counts = project(velocity_cfg, velocity=0.49, previous_acceleration=0.4, expected_infeasible=True)
    assert value == pytest.approx(-0.2, abs=1.0e-5)
    assert counts[2] == 1

    position_cfg = _unconstrained_cfg(
        dt=0.1,
        max_joint_velocity=(100.0,) * _NUM_JOINTS,
        max_joint_acceleration=(100.0,) * _NUM_JOINTS,
        max_joint_jerk=(1000.0,) * _NUM_JOINTS,
    )
    value, counts = project(position_cfg, position=0.895)
    assert value == pytest.approx(3.0, abs=2.0e-5)
    assert counts[3] == 1

    value, counts = project(position_cfg, position=0.895, previous_acceleration=0.6)
    assert value == pytest.approx(1.8, abs=2.0e-5)
    assert counts[3] == 1

    combined_cfg = _unconstrained_cfg(
        dt=0.1,
        max_joint_velocity=(0.5,) * _NUM_JOINTS,
        max_joint_acceleration=(1.0,) * _NUM_JOINTS,
        max_joint_jerk=(2.0,) * _NUM_JOINTS,
    )
    value, counts = project(combined_cfg, velocity=0.49)
    assert value == pytest.approx(0.2)
    assert counts[:3] == (1, 1, 1)


def test_endpoint_only_projection_misses_240hz_overshoot_but_continuous_envelope_faults():
    cfg = _unconstrained_cfg(
        dt=0.1,
        max_joint_velocity=(0.5,) * _NUM_JOINTS,
        max_joint_acceleration=(100.0,) * _NUM_JOINTS,
        max_joint_jerk=(1000.0,) * _NUM_JOINTS,
    )
    controller = _make_controller(cfg=cfg, lower=-1.0, upper=1.0)
    position = torch.zeros(1, _NUM_JOINTS)
    velocity = torch.zeros_like(position)
    velocity[0, 0] = 0.49
    controller.reset(position, velocity)
    controller._joint_acceleration[0, 0] = 0.4
    nominal = torch.zeros_like(position)

    projected, infeasible, _, _, velocity_count, _ = controller._project_acceleration(nominal)

    assert projected[0, 0].item() == pytest.approx(-0.2, abs=1.0e-5)
    assert infeasible.item()
    assert velocity_count.item() == 1

    # This is the command accepted by the previous endpoint-only projection:
    # v(0)=0.49 and v(T)=0.50, but the quadratic reaches 0.5033 rad/s
    # between them. Sampling every 1/240 s catches the unsafe interior peak.
    tracker_times = torch.arange(25, dtype=torch.float64) / 240.0
    old_endpoint_acceleration = projected[0, 0].double()
    constant_jerk = (old_endpoint_acceleration - 0.4) / cfg.dt
    sampled_velocity = 0.49 + tracker_times * 0.4 + 0.5 * tracker_times.square() * constant_jerk
    assert sampled_velocity[-1].item() == pytest.approx(0.5, abs=1.0e-7)
    assert sampled_velocity.max().item() > 0.503

    held_position = controller.joint_position.clone()
    held_velocity = controller.joint_velocity.clone()
    controller.step(torch.zeros(1, 6), _zero_kinematics(controller))
    assert controller.telemetry.infeasible.item()
    assert controller.faulted.item()
    torch.testing.assert_close(controller.joint_position, held_position)
    torch.testing.assert_close(controller.joint_velocity, held_velocity)


def test_position_bernstein_control_point_violation_is_infeasible():
    cfg = _unconstrained_cfg(
        dt=0.1,
        max_joint_velocity=(100.0,) * _NUM_JOINTS,
        max_joint_acceleration=(100.0,) * _NUM_JOINTS,
        max_joint_jerk=(1000.0,) * _NUM_JOINTS,
    )
    controller = _make_controller(cfg=cfg, lower=-1.0, upper=1.0)
    position = torch.zeros(1, _NUM_JOINTS)
    velocity = torch.zeros_like(position)
    position[0, 0] = 0.895
    velocity[0, 0] = 0.1
    controller.reset(position, velocity)
    controller._joint_acceleration[0, 0] = 0.6

    _, infeasible, _, _, _, position_count = controller._project_acceleration(torch.zeros_like(position))

    # q0 + 2*T*v0/3 + T^2*a0/6 = 0.902667 exceeds the
    # safe upper bound 0.9 even though a suitable a1 can bound q1.
    assert infeasible.item()
    assert position_count.item() == 1


def test_infeasible_bound_fault_holds_persistently_and_selective_reset_is_isolated():
    cfg = _unconstrained_cfg(
        dt=0.1,
        wrench_scale=(1.0,) * 6,
        max_joint_velocity=(0.5,) * _NUM_JOINTS,
        max_joint_acceleration=(1.0,) * _NUM_JOINTS,
        max_joint_jerk=(0.1,) * _NUM_JOINTS,
    )
    controller = _make_controller(num_envs=3, cfg=cfg, lower=-1.0, upper=1.0)
    _reset_zero(controller)
    unsafe_position = controller.safe_joint_upper_limits[1].reshape(1, -1).clone()
    outward_velocity = torch.full((1, _NUM_JOINTS), 0.5)
    controller.reset(unsafe_position, outward_velocity, env_ids=[1])
    held_position = controller.joint_position[1].clone()
    held_velocity = controller.joint_velocity[1].clone()
    kinematics = _zero_kinematics(controller)
    kinematics.tcp_jacobian[:, 0, 0] = 1.0
    action = torch.zeros(3, 6)
    action[0, 0] = 0.1
    action[2, 0] = -0.1

    controller.step(action, kinematics)
    assert controller.telemetry.infeasible[1].item()
    assert controller.faulted[1].item()
    torch.testing.assert_close(controller.joint_position[1], held_position)
    torch.testing.assert_close(controller.joint_velocity[1], held_velocity)

    controller.step(action, kinematics)
    assert controller.faulted[1].item()
    torch.testing.assert_close(controller.joint_position[1], held_position)
    torch.testing.assert_close(controller.joint_velocity[1], held_velocity)

    neighbor_position = controller.joint_position[[0, 2]].clone()
    neighbor_velocity = controller.joint_velocity[[0, 2]].clone()
    neighbor_metric_telemetry = controller.telemetry.metric_min_cholesky_diagonal[[0, 2]].clone()
    controller.reset(torch.zeros(1, _NUM_JOINTS), env_ids=[1])

    assert not controller.faulted[1].item()
    torch.testing.assert_close(controller.joint_position[[0, 2]], neighbor_position)
    torch.testing.assert_close(controller.joint_velocity[[0, 2]], neighbor_velocity)
    torch.testing.assert_close(controller.telemetry.metric_min_cholesky_diagonal[[0, 2]], neighbor_metric_telemetry)
    for telemetry_field in fields(controller.telemetry):
        assert torch.count_nonzero(getattr(controller.telemetry, telemetry_field.name)[1]).item() == 0


def test_nonfinite_and_cholesky_failures_fault_without_throwing():
    controller = _make_controller(num_envs=2)
    _reset_zero(controller)
    kinematics = _zero_kinematics(controller)
    kinematics.transverse_jacobian[0, 0, 0] = torch.inf
    initial_position = controller.joint_position.clone()

    controller.step(torch.zeros(2, 6), kinematics)

    assert controller.telemetry.nonfinite[0].item()
    assert controller.telemetry.cholesky_failed[0].item()
    assert controller.faulted[0].item()
    assert not controller.faulted[1].item()
    torch.testing.assert_close(controller.joint_position[0], initial_position[0])
    for telemetry_field in fields(controller.telemetry):
        value = getattr(controller.telemetry, telemetry_field.name)
        if torch.is_floating_point(value):
            assert torch.all(torch.isfinite(value))

    valid_kinematics = _zero_kinematics(controller)
    controller.step(torch.zeros(2, 6), valid_kinematics)
    assert controller.faulted[0].item()
    torch.testing.assert_close(controller.joint_position[0], initial_position[0])

    indefinite = _make_controller()
    _reset_zero(indefinite)
    indefinite._root_metric[0, 0] = -1.0
    indefinite.step(torch.zeros(1, 6), _zero_kinematics(indefinite))
    assert indefinite.telemetry.cholesky_failed.item()
    assert indefinite.faulted.item()


def test_reset_rejects_an_unsafe_posture_target():
    controller = _make_controller()
    position = torch.zeros(1, _NUM_JOINTS)
    posture = position.clone()
    posture[0, 0] = controller.safe_joint_upper_limits[0, 0] + 0.01

    with pytest.raises(ValueError, match="Posture targets"):
        controller.reset(position, posture_target=posture)


def test_reset_rejects_nonfinite_state():
    controller = _make_controller()
    position = torch.zeros(1, _NUM_JOINTS)
    velocity = torch.zeros_like(position)
    position[0, 0] = torch.nan
    with pytest.raises(ValueError, match="positions must be finite"):
        controller.reset(position, velocity)

    position[0, 0] = 0.0
    velocity[0, 0] = torch.inf
    with pytest.raises(ValueError, match="velocities must be finite"):
        controller.reset(position, velocity)


def _run_trace(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    controller = _make_controller(num_envs=8, device=device)
    generator = torch.Generator().manual_seed(29)
    position = 0.2 * (torch.rand(8, _NUM_JOINTS, generator=generator) - 0.5)
    velocity = 0.08 * (torch.rand(8, _NUM_JOINTS, generator=generator) - 0.5)
    error = 0.02 * (torch.rand(8, 2, generator=generator) - 0.5)
    transverse_jacobian = 0.2 * torch.randn(8, 2, _NUM_JOINTS, generator=generator)
    jacobian_dot_velocity = 0.02 * torch.randn(8, 2, generator=generator)
    tcp_jacobian = 0.2 * torch.randn(8, 6, _NUM_JOINTS, generator=generator)
    actions = [0.5 * torch.randn(8, 6, generator=generator) for _ in range(4)]

    controller.reset(position.to(device), velocity.to(device))
    kinematics = JointFabricTaskKinematics(
        transverse_error=error.to(device),
        transverse_jacobian=transverse_jacobian.to(device),
        transverse_jacobian_dot_velocity=jacobian_dot_velocity.to(device),
        tcp_jacobian=tcp_jacobian.to(device),
    )
    for action in actions:
        controller.step(action.to(device), kinematics)
    return (
        controller.joint_position.cpu(),
        controller.joint_velocity.cpu(),
        controller.joint_acceleration.cpu(),
        controller.telemetry.energy_residual.cpu(),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_and_cuda_traces_match():
    cpu_trace = _run_trace("cpu")
    cuda_trace = _run_trace("cuda")

    for cpu_value, cuda_value in zip(cpu_trace, cuda_trace, strict=True):
        torch.testing.assert_close(cpu_value, cuda_value, rtol=2.0e-4, atol=2.0e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_graph_replay_matches_eager_execution():
    cfg = _unconstrained_cfg(dt=0.01, wrench_scale=(1.0,) * 6)
    captured = _make_controller(device="cuda", cfg=cfg)
    eager = _make_controller(device="cuda", cfg=cfg)
    initial_position = torch.zeros(1, _NUM_JOINTS, device="cuda")
    captured.reset(initial_position)
    eager.reset(initial_position)
    kinematics = _zero_kinematics(captured)
    kinematics.tcp_jacobian[0, 0, 0] = 1.0
    eager_kinematics = JointFabricTaskKinematics(
        transverse_error=kinematics.transverse_error.clone(),
        transverse_jacobian=kinematics.transverse_jacobian.clone(),
        transverse_jacobian_dot_velocity=kinematics.transverse_jacobian_dot_velocity.clone(),
        tcp_jacobian=kinematics.tcp_jacobian.clone(),
    )
    static_action = torch.zeros(1, 6, device="cuda")

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        captured.step(static_action, kinematics)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    captured.reset(initial_position)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured.step(static_action, kinematics)
    eager.step(static_action, eager_kinematics)

    static_action[0, 0] = 0.4
    graph.replay()
    eager.step(static_action, eager_kinematics)
    torch.cuda.synchronize()

    torch.testing.assert_close(captured.joint_position, eager.joint_position, rtol=1.0e-5, atol=1.0e-7)
    torch.testing.assert_close(captured.joint_velocity, eager.joint_velocity, rtol=1.0e-5, atol=1.0e-7)
    torch.testing.assert_close(captured.joint_acceleration, eager.joint_acceleration, rtol=1.0e-5, atol=1.0e-7)
