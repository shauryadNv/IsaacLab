# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration regressions for the isolated DisplayPort fabric experiment."""

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

from isaaclab.managers import ObservationTermCfg

from isaaclab_tasks.contrib.deploy.cable_insertion.config import displayport_rizon_4s  # noqa: F401
from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s import fabric_terminations
from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.fabric_actions import (
    DisplayportFabricAction,
    _sample_constant_jerk_segment,
    _socket_transverse_error,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.fabric_actions_cfg import (
    DisplayportFabricActionCfg,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.fabric_env_cfg import (
    _FABRIC_SOCKET_TCP_GOAL_OFFSET,
    FabricObservationsCfg,
    Rizon4sFabricDisplayportInsertionEnvCfg,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.joint_fabric_controller import (
    JointFabricController,
    JointFabricControllerCfg,
)
from isaaclab_tasks.contrib.deploy.mdp.actions_cfg import DeployOperationalSpaceControllerActionCfg

_SHIPPING_TASKS = {
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav": (
        "joint_pos_env_cfg:Rizon4sGravDisplayportInsertionEnvCfg"
    ),
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Play": (
        "joint_pos_env_cfg:Rizon4sGravDisplayportInsertionEnvCfg_PLAY"
    ),
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel": (
        "joint_pos_env_cfg:Rizon4sGravDisplayportInsertionNoJointVelEnvCfg"
    ),
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-Play": (
        "joint_pos_env_cfg:Rizon4sGravDisplayportInsertionNoJointVelEnvCfg_PLAY"
    ),
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-ROS-Inference": (
        "ros_inference_env_cfg:Rizon4sGravDisplayportInsertionNoJointVelROSInferenceEnvCfg"
    ),
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-ROS-Inference": (
        "ros_inference_env_cfg:Rizon4sGravDisplayportInsertionROSInferenceEnvCfg"
    ),
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace": (
        "task_space_env_cfg:Rizon4sTaskSpaceDisplayportInsertionEnvCfg"
    ),
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Play": (
        "task_space_env_cfg:Rizon4sTaskSpaceDisplayportInsertionEnvCfg_PLAY"
    ),
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-ROS-Inference": (
        "task_space_ros_inference_env_cfg:Rizon4sTaskSpaceDisplayportInsertionROSInferenceEnvCfg"
    ),
}


def _observation_term_names(group) -> list[str]:
    return [name for name, value in vars(group).items() if isinstance(value, ObservationTermCfg)]


def test_existing_task_registrations_are_unchanged():
    expected_entry_point = "isaaclab_tasks.contrib.deploy.cable_insertion.insertion_env:DisplayportInsertionEnv"
    package = "isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s"
    for task_id, cfg_suffix in _SHIPPING_TASKS.items():
        spec = gym.spec(task_id)
        assert spec.entry_point == expected_entry_point
        assert spec.kwargs["env_cfg_entry_point"] == f"{package}.{cfg_suffix}"


def test_fabric_tasks_use_isolated_entrypoint_and_config():
    package = "isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s"
    for suffix, cfg_class in (
        ("", "Rizon4sFabricDisplayportInsertionEnvCfg"),
        ("-Play", "Rizon4sFabricDisplayportInsertionEnvCfg_PLAY"),
    ):
        spec = gym.spec(f"IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Fabric{suffix}")
        assert spec.entry_point == f"{package}.fabric_insertion_env:DisplayportFabricInsertionEnv"
        assert spec.kwargs["env_cfg_entry_point"] == f"{package}.fabric_env_cfg:{cfg_class}"
        assert (
            "fabric_rsl_rl_ppo_cfg:Rizon4sFabricDisplayportInsertionRNNPPORunnerCfg"
            in spec.kwargs["rsl_rl_cfg_entry_point"]
        )


def test_fabric_config_is_physx_explicit_pd_and_rate_consistent():
    cfg = Rizon4sFabricDisplayportInsertionEnvCfg()
    cfg.validate()

    assert isinstance(cfg.actions.arm_action, DisplayportFabricActionCfg)
    assert cfg.sim.dt == pytest.approx(1.0 / 240.0)
    assert cfg.decimation == 8
    assert cfg.actions.arm_action.fabric_cfg.dt == pytest.approx(1.0 / 60.0)
    assert cfg.decimation * cfg.sim.dt == pytest.approx(1.0 / 30.0)
    assert cfg.actions.arm_action.use_inertial_feedforward
    assert cfg.actions.arm_action.use_gravity_compensation
    assert not cfg.scene.robot.spawn.rigid_props.disable_gravity
    assert cfg.terminations.fabric_faulted.func is fabric_terminations.fabric_faulted

    original = displayport_rizon_4s.task_space_env_cfg.Rizon4sTaskSpaceDisplayportInsertionEnvCfg()
    assert isinstance(original.actions.arm_action, DeployOperationalSpaceControllerActionCfg)
    assert original.scene.robot.spawn.rigid_props.disable_gravity


def test_fabric_observation_order_is_stable():
    observations = FabricObservationsCfg()
    assert _observation_term_names(observations.policy) == [
        "eef_pos",
        "eef_rot_6d",
        "socket_kp_pos",
        "socket_kp_rot_6d",
        "joint_pos",
        "joint_vel",
        "fabric_joint_pos",
        "fabric_joint_vel",
    ]
    assert _observation_term_names(observations.critic) == [
        "joint_pos",
        "joint_vel",
        "fabric_joint_pos",
        "fabric_joint_vel",
        "socket_kp_pos",
        "socket_kp_rot_6d",
        "plug_kp_pos",
        "plug_kp_rot_6d",
    ]


def test_arm_state_observations_preserve_configured_joint_order():
    observations = FabricObservationsCfg()
    for group in (observations.policy, observations.critic):
        assert group.joint_pos.params["asset_cfg"].preserve_order
        assert group.joint_vel.params["asset_cfg"].preserve_order


def test_constant_jerk_tracker_samples_are_exact_c2_and_jerk_bounded():
    duration = 1.0 / 60.0
    tracker_dt = 1.0 / 240.0
    position_0 = torch.tensor([[0.1, -0.2]], dtype=torch.float64)
    velocity_0 = torch.tensor([[0.3, -0.1]], dtype=torch.float64)
    acceleration_0 = torch.tensor([[0.2, -0.4]], dtype=torch.float64)
    acceleration_1 = torch.tensor([[0.3, -0.2]], dtype=torch.float64)

    samples = [
        _sample_constant_jerk_segment(
            position_0,
            velocity_0,
            acceleration_0,
            acceleration_1,
            duration,
            tick * tracker_dt,
        )
        for tick in range(5)
    ]
    position_1, velocity_1, sampled_acceleration_1 = samples[-1]
    torch.testing.assert_close(sampled_acceleration_1, acceleration_1)
    torch.testing.assert_close(
        velocity_1,
        velocity_0 + 0.5 * duration * (acceleration_0 + acceleration_1),
    )
    torch.testing.assert_close(
        position_1,
        position_0 + duration * velocity_0 + duration**2 * (acceleration_0 / 3.0 + acceleration_1 / 6.0),
    )
    assert not torch.equal(samples[1][0], position_0)

    sampled_accelerations = torch.stack([sample[2] for sample in samples], dim=0)
    sampled_jerk = torch.diff(sampled_accelerations, dim=0) / tracker_dt
    segment_jerk = (acceleration_1 - acceleration_0) / duration
    torch.testing.assert_close(sampled_jerk, segment_jerk.expand_as(sampled_jerk))

    acceleration_2 = torch.tensor([[-0.1, 0.1]], dtype=torch.float64)
    next_start = _sample_constant_jerk_segment(
        position_1,
        velocity_1,
        acceleration_1,
        acceleration_2,
        duration,
        0.0,
    )
    torch.testing.assert_close(next_start[0], position_1)
    torch.testing.assert_close(next_start[1], velocity_1)
    torch.testing.assert_close(next_start[2], acceleration_1)
    next_first_tracker_tick = _sample_constant_jerk_segment(
        position_1,
        velocity_1,
        acceleration_1,
        acceleration_2,
        duration,
        tracker_dt,
    )
    boundary_jerk = (next_first_tracker_tick[2] - sampled_acceleration_1) / tracker_dt
    torch.testing.assert_close(boundary_jerk, (acceleration_2 - acceleration_1) / duration)


def test_every_240hz_sample_of_certified_constant_jerk_segment_stays_inside_envelope():
    fabric_dt = 0.1
    controller_cfg = JointFabricControllerCfg(
        dt=fabric_dt,
        max_joint_velocity=(0.5,) * 7,
        max_joint_acceleration=(100.0,) * 7,
        max_joint_jerk=(1000.0,) * 7,
    )
    controller = JointFabricController(
        controller_cfg,
        num_envs=1,
        device="cpu",
        joint_lower_limits=(-1.0,) * 7,
        joint_upper_limits=(1.0,) * 7,
    )
    position_0 = torch.zeros(1, 7)
    velocity_0 = torch.zeros(1, 7)
    acceleration_0 = torch.zeros(1, 7)
    acceleration_1 = torch.zeros(1, 7)
    velocity_0[0, 0] = 0.45
    acceleration_0[0, 0] = 0.2
    acceleration_1[0, 0] = -0.2
    controller.reset(position_0, velocity_0)
    controller._joint_acceleration[:] = acceleration_0
    _, infeasible, *_ = controller._project_acceleration(acceleration_1)
    assert not infeasible.item()

    action = object.__new__(DisplayportFabricAction)
    action._controller = controller
    action.cfg = SimpleNamespace(fabric_cfg=controller.cfg)
    action._policy_sum = {
        "tracker_position_limit_violation_count": torch.zeros(1, dtype=torch.long),
        "tracker_velocity_limit_violation_count": torch.zeros(1, dtype=torch.long),
    }
    action._policy_or = {
        "faulted": torch.zeros(1, dtype=torch.bool),
        "nonfinite": torch.zeros(1, dtype=torch.bool),
    }

    for tick in range(1, 25):
        position, velocity, acceleration = _sample_constant_jerk_segment(
            position_0,
            velocity_0,
            acceleration_0,
            acceleration_1,
            fabric_dt,
            tick / 240.0,
        )
        guarded_position, guarded_velocity, _ = action._guard_tracker_targets(
            position, velocity, acceleration, position_0
        )
        torch.testing.assert_close(guarded_position, position)
        torch.testing.assert_close(guarded_velocity, velocity)

    assert not controller.faulted.item()
    assert action._policy_sum["tracker_position_limit_violation_count"].item() == 0
    assert action._policy_sum["tracker_velocity_limit_violation_count"].item() == 0


def test_tracker_sample_limit_guard_faults_and_holds_out_of_envelope_targets():
    controller = JointFabricController(
        JointFabricControllerCfg(),
        num_envs=2,
        device="cpu",
        joint_lower_limits=(-1.0,) * 7,
        joint_upper_limits=(1.0,) * 7,
    )
    controller.reset(torch.zeros(2, 7))
    action = object.__new__(DisplayportFabricAction)
    action._controller = controller
    action.cfg = SimpleNamespace(fabric_cfg=controller.cfg)
    action._policy_sum = {
        "tracker_position_limit_violation_count": torch.zeros(2, dtype=torch.long),
        "tracker_velocity_limit_violation_count": torch.zeros(2, dtype=torch.long),
    }
    action._policy_or = {
        "faulted": torch.zeros(2, dtype=torch.bool),
        "nonfinite": torch.zeros(2, dtype=torch.bool),
    }

    position_target = torch.zeros(2, 7)
    velocity_target = torch.zeros_like(position_target)
    acceleration_target = torch.zeros_like(position_target)
    position_target[1, 0] = controller.safe_joint_upper_limits[1, 0] + 0.01
    velocity_target[1, 1] = controller.maximum_joint_velocity[0, 1] + 0.01
    measured_position = torch.full_like(position_target, 0.2)

    guarded_position, guarded_velocity, guarded_acceleration = action._guard_tracker_targets(
        position_target, velocity_target, acceleration_target, measured_position
    )

    assert not controller.faulted[0].item()
    assert controller.faulted[1].item()
    torch.testing.assert_close(guarded_position[0], position_target[0])
    torch.testing.assert_close(guarded_position[1], measured_position[1])
    torch.testing.assert_close(guarded_velocity[1], torch.zeros(7))
    torch.testing.assert_close(guarded_acceleration[1], torch.zeros(7))
    torch.testing.assert_close(action._policy_sum["tracker_position_limit_violation_count"], torch.tensor([0, 1]))
    torch.testing.assert_close(action._policy_sum["tracker_velocity_limit_violation_count"], torch.tensor([0, 1]))
    assert action._policy_or["faulted"][1].item()


def test_measured_motion_and_actual_effort_telemetry_is_reset_baselined_and_faults_nonfinite():
    controller = JointFabricController(
        JointFabricControllerCfg(),
        num_envs=2,
        device="cpu",
        joint_lower_limits=(-1.0,) * 7,
        joint_upper_limits=(1.0,) * 7,
    )
    controller.reset(torch.zeros(2, 7))
    action = object.__new__(DisplayportFabricAction)
    action._controller = controller
    action._joint_ids = list(range(7))
    action._env = SimpleNamespace(physics_dt=0.25)
    actual_effort = torch.zeros(2, 7)
    action._asset = SimpleNamespace(actuators=SimpleNamespace(applied_effort=SimpleNamespace(torch=actual_effort)))
    action._policy_or = {
        "faulted": torch.zeros(2, dtype=torch.bool),
        "nonfinite": torch.zeros(2, dtype=torch.bool),
    }
    action._policy_max = {
        name: torch.zeros(2)
        for name in (
            "maximum_measured_joint_acceleration",
            "maximum_measured_joint_jerk",
            "maximum_actual_applied_effort",
            "maximum_actual_applied_effort_slew",
        )
    }
    action._segment_position_0 = torch.zeros(2, 7)
    action._previous_measured_velocity = torch.zeros(2, 7)
    action._previous_measured_acceleration = torch.zeros(2, 7)
    action._previous_actual_applied_effort = torch.zeros(2, 7)
    action._tracker_history_valid = torch.zeros(2, dtype=torch.bool)
    indices = torch.arange(2)
    measured_position = torch.zeros(2, 7)
    initial_velocity = torch.zeros(2, 7)
    action._reset_tracker_histories(indices, initial_velocity, actual_effort.clone())

    measured_velocity = torch.stack((torch.full((7,), 0.25), torch.full((7,), -0.5)))
    actual_effort[:] = torch.stack((torch.full((7,), 0.5), torch.full((7,), -1.0)))
    safe_position, safe_velocity = action._update_tracker_health_telemetry(measured_position, measured_velocity)

    torch.testing.assert_close(safe_position, measured_position)
    torch.testing.assert_close(safe_velocity, measured_velocity)
    torch.testing.assert_close(action._policy_max["maximum_measured_joint_acceleration"], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(action._policy_max["maximum_measured_joint_jerk"], torch.tensor([4.0, 8.0]))
    torch.testing.assert_close(action._policy_max["maximum_actual_applied_effort"], torch.tensor([0.5, 1.0]))
    torch.testing.assert_close(action._policy_max["maximum_actual_applied_effort_slew"], torch.tensor([2.0, 4.0]))
    assert not torch.any(controller.faulted)

    action._reset_tracker_histories(indices, measured_velocity, actual_effort.clone())
    for value in action._policy_max.values():
        value.zero_()
    action._update_tracker_health_telemetry(measured_position, measured_velocity)
    torch.testing.assert_close(action._policy_max["maximum_measured_joint_acceleration"], torch.zeros(2))
    torch.testing.assert_close(action._policy_max["maximum_measured_joint_jerk"], torch.zeros(2))
    torch.testing.assert_close(action._policy_max["maximum_actual_applied_effort_slew"], torch.zeros(2))

    actual_effort[1, 0] = torch.nan
    safe_position, safe_velocity = action._update_tracker_health_telemetry(measured_position, measured_velocity)
    assert controller.faulted[1].item()
    assert action._policy_or["nonfinite"][1].item()
    assert torch.all(torch.isfinite(safe_position))
    assert torch.all(torch.isfinite(safe_velocity))


def test_policy_telemetry_aggregates_and_survives_accumulator_reset():
    action = object.__new__(DisplayportFabricAction)
    action._policy_or = {
        name: torch.zeros(2, dtype=torch.bool) for name in ("faulted", "nonfinite", "infeasible", "cholesky_failed")
    }
    action._policy_sum = {
        name: torch.zeros(2, dtype=torch.long)
        for name in (
            "action_saturation_count",
            "acceleration_bound_active_count",
            "jerk_bound_active_count",
            "velocity_bound_active_count",
            "position_bound_active_count",
            "roundoff_guard_active_count",
            "tracker_effort_saturation_count",
            "tracker_position_limit_violation_count",
            "tracker_velocity_limit_violation_count",
        )
    }
    action._policy_max = {
        name: torch.zeros(2)
        for name in (
            "tracking_position_error_norm",
            "tracking_velocity_error_norm",
            "maximum_acceleration",
            "maximum_jerk",
            "solve_residual_norm",
            "predicted_tracker_computed_effort",
            "predicted_tracker_applied_effort",
            "maximum_measured_joint_acceleration",
            "maximum_measured_joint_jerk",
            "maximum_actual_applied_effort",
            "maximum_actual_applied_effort_slew",
        )
    }
    action._policy_min = {
        name: torch.full((2,), torch.inf) for name in ("minimum_joint_margin", "metric_min_cholesky_diagonal")
    }
    telemetry_buffers = {}
    for values in (action._policy_or, action._policy_sum, action._policy_max, action._policy_min):
        telemetry_buffers.update({name: torch.zeros_like(value) for name, value in values.items()})
    action._policy_telemetry = SimpleNamespace(**telemetry_buffers)

    def telemetry(*, faulted, count, acceleration, residual, margin, cholesky):
        return SimpleNamespace(
            faulted=torch.tensor(faulted),
            nonfinite=torch.tensor([False, False]),
            infeasible=torch.tensor([False, False]),
            cholesky_failed=torch.tensor([False, False]),
            action_saturation_count=torch.tensor(count),
            acceleration_bound_active_count=torch.tensor(count),
            jerk_bound_active_count=torch.tensor(count),
            velocity_bound_active_count=torch.tensor(count),
            position_bound_active_count=torch.tensor(count),
            roundoff_guard_active_count=torch.tensor(count),
            maximum_acceleration=torch.tensor(acceleration),
            maximum_jerk=torch.tensor(acceleration),
            solve_residual_norm=torch.tensor(residual),
            minimum_joint_margin=torch.tensor(margin),
            metric_min_cholesky_diagonal=torch.tensor(cholesky),
        )

    action._aggregate_controller_telemetry(
        telemetry(
            faulted=[True, False],
            count=[1, 2],
            acceleration=[1.0, 2.0],
            residual=[0.1, 0.2],
            margin=[0.4, 0.3],
            cholesky=[0.8, 0.7],
        )
    )
    action._aggregate_controller_telemetry(
        telemetry(
            faulted=[False, True],
            count=[3, 4],
            acceleration=[3.0, 1.0],
            residual=[0.05, 0.4],
            margin=[0.2, 0.5],
            cholesky=[0.6, 0.9],
        )
    )
    action._policy_sum["tracker_effort_saturation_count"][:] = torch.tensor([5, 6])
    action._policy_sum["tracker_position_limit_violation_count"][:] = torch.tensor([0, 1])
    action._policy_sum["tracker_velocity_limit_violation_count"][:] = torch.tensor([2, 0])
    action._policy_max["predicted_tracker_computed_effort"][:] = torch.tensor([20.0, 30.0])
    action._policy_max["predicted_tracker_applied_effort"][:] = torch.tensor([10.0, 11.0])
    action._policy_max["maximum_measured_joint_acceleration"][:] = torch.tensor([1.0, 2.0])
    action._policy_max["maximum_measured_joint_jerk"][:] = torch.tensor([3.0, 4.0])
    action._policy_max["maximum_actual_applied_effort"][:] = torch.tensor([5.0, 6.0])
    action._policy_max["maximum_actual_applied_effort_slew"][:] = torch.tensor([7.0, 8.0])
    action._finalize_policy_telemetry()
    snapshot = action.policy_telemetry

    assert torch.all(snapshot.faulted)
    torch.testing.assert_close(snapshot.action_saturation_count, torch.tensor([4, 6]))
    torch.testing.assert_close(snapshot.maximum_acceleration, torch.tensor([3.0, 2.0]))
    torch.testing.assert_close(snapshot.solve_residual_norm, torch.tensor([0.1, 0.4]))
    torch.testing.assert_close(snapshot.minimum_joint_margin, torch.tensor([0.2, 0.3]))
    torch.testing.assert_close(snapshot.metric_min_cholesky_diagonal, torch.tensor([0.6, 0.7]))
    torch.testing.assert_close(snapshot.tracker_position_limit_violation_count, torch.tensor([0, 1]))
    torch.testing.assert_close(snapshot.tracker_velocity_limit_violation_count, torch.tensor([2, 0]))
    torch.testing.assert_close(snapshot.maximum_measured_joint_acceleration, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(snapshot.maximum_measured_joint_jerk, torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(snapshot.maximum_actual_applied_effort, torch.tensor([5.0, 6.0]))
    torch.testing.assert_close(snapshot.maximum_actual_applied_effort_slew, torch.tensor([7.0, 8.0]))

    action._reset_policy_telemetry_accumulators()
    assert torch.all(snapshot.faulted)
    torch.testing.assert_close(snapshot.action_saturation_count, torch.tensor([4, 6]))


def test_fault_termination_reads_preserved_policy_snapshot():
    faulted = torch.tensor([False, True, False])
    fake_action = SimpleNamespace(policy_telemetry=SimpleNamespace(faulted=faulted))
    fake_env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda _: fake_action))
    assert fabric_terminations.fabric_faulted(fake_env) is faulted


def test_seated_tcp_goal_has_zero_transverse_error_and_x_is_unconstrained():
    goal = torch.tensor(_FABRIC_SOCKET_TCP_GOAL_OFFSET, dtype=torch.float64).reshape(1, 3)
    torch.testing.assert_close(goal, torch.tensor([[0.0171, 0.0, 0.0025]], dtype=torch.float64), atol=2.0e-6, rtol=0.0)
    torch.testing.assert_close(_socket_transverse_error(goal, goal), torch.zeros(1, 2, dtype=torch.float64))

    displaced_only_along_insertion = goal + torch.tensor([[0.25, 0.0, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(
        _socket_transverse_error(displaced_only_along_insertion, goal),
        torch.zeros(1, 2, dtype=torch.float64),
    )


def test_invalid_policy_rate_is_rejected():
    cfg = Rizon4sFabricDisplayportInsertionEnvCfg()
    cfg.decimation = 7
    with pytest.raises(ValueError, match="policy decimation"):
        cfg.validate()


def test_substep_fabric_rate_is_rejected():
    cfg = Rizon4sFabricDisplayportInsertionEnvCfg()
    cfg.actions.arm_action.fabric_cfg.dt = 0.1 * cfg.sim.dt
    with pytest.raises(ValueError, match="at least one simulation step"):
        cfg.validate()
