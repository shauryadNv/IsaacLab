# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wiring that turns a :class:`DomainRandCfg` into event, observation and action terms.

This module owns the single mapping from each configuration knob to the term it drives
and to the address the ADR curriculum interpolates, so the two can never drift apart.
"""

from __future__ import annotations

__all__ = ["apply_domain_randomization"]

from collections.abc import Callable
from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg, UniformNoiseCfg

import isaaclab_tasks.contrib.deploy.mdp as mdp

from .domain_rand_cfg import AxisKnobCfg, DomainRandCfg, ScalarKnobCfg, SignalNoiseKnobCfg

if TYPE_CHECKING:
    from .task_space_env_cfg import Rizon4sTaskSpaceDisplayportInsertionEnvCfg

# Policy observation term backing each observation-noise knob.
_OBS_TERMS = {
    "obs_eef_pos": "eef_pos",
    "obs_eef_rot": "eef_rot_6d",
    "obs_socket_pos": "socket_kp_pos",
    "obs_socket_rot": "socket_kp_rot_6d",
}

_AXIS_KEYS = ("x", "y", "z")
_ROT_KEYS = ("roll", "pitch", "yaw")

# Registers a range for the ADR curriculum to widen: (address, initial, final).
TrackFn = Callable[[str, object, object], None]


def _symmetric_range(halfwidths, keys) -> dict[str, list[float]]:
    """Turn per-axis half-widths into the ``{axis: [-h, h]}`` form event terms expect."""
    return {key: [-float(h), float(h)] for key, h in zip(keys, halfwidths)}


def _pose_range(pos: AxisKnobCfg | tuple, rot: AxisKnobCfg | tuple) -> dict[str, list[float]]:
    pos_values = pos.initial if isinstance(pos, AxisKnobCfg) else pos
    rot_values = rot.initial if isinstance(rot, AxisKnobCfg) else rot
    return {**_symmetric_range(pos_values, _AXIS_KEYS), **_symmetric_range(rot_values, _ROT_KEYS)}


def _apply_controller_gains(dr: DomainRandCfg, events, track: TrackFn) -> None:
    """Randomize the OSC task-space gains.

    The task-space env zeroes the arm joint PD so the controller can drive the joints with
    pure torque, which makes joint-gain randomization a no-op. These are the gains that
    are actually in the loop.
    """
    if not (dr.osc_stiffness.enable or dr.osc_damping_ratio.enable):
        return
    events.randomize_osc_gains = EventTerm(
        func=mdp.randomize_osc_task_gains,
        mode="reset",
        params={
            "stiffness_scale_range": tuple(dr.osc_stiffness.initial),
            "damping_ratio_scale_range": tuple(dr.osc_damping_ratio.initial),
            "action_term_name": "arm_action",
        },
    )
    if dr.osc_stiffness.enable:
        track(
            "events.randomize_osc_gains.params.stiffness_scale_range",
            tuple(dr.osc_stiffness.initial),
            tuple(dr.osc_stiffness.final),
        )
    if dr.osc_damping_ratio.enable:
        track(
            "events.randomize_osc_gains.params.damping_ratio_scale_range",
            tuple(dr.osc_damping_ratio.initial),
            tuple(dr.osc_damping_ratio.final),
        )


def _apply_joint_properties(dr: DomainRandCfg, env_cfg, events, track: TrackFn) -> None:
    """Randomize arm joint armature and friction.

    Both act on the plant, which the decoupling-off OSC law never models, so they create a
    genuine plant/controller mismatch. The Flexiv asset ships armature = friction = 0.0, so
    these must be absolute values; a multiplicative scale would be a no-op.
    """
    if not (dr.joint_armature.enable or dr.joint_friction.enable):
        return
    params = {
        "asset_cfg": SceneEntityCfg("robot", joint_names=list(env_cfg.actions.arm_action.joint_names)),
        "operation": "abs",
        "distribution": "uniform",
    }
    if dr.joint_armature.enable:
        params["armature_distribution_params"] = tuple(dr.joint_armature.initial)
    if dr.joint_friction.enable:
        params["friction_distribution_params"] = tuple(dr.joint_friction.initial)
    events.randomize_joint_props = EventTerm(func=mdp.randomize_joint_parameters, mode="reset", params=params)
    if dr.joint_armature.enable:
        track(
            "events.randomize_joint_props.params.armature_distribution_params",
            tuple(dr.joint_armature.initial),
            tuple(dr.joint_armature.final),
        )
    if dr.joint_friction.enable:
        track(
            "events.randomize_joint_props.params.friction_distribution_params",
            tuple(dr.joint_friction.initial),
            tuple(dr.joint_friction.final),
        )


def _apply_contact_properties(dr: DomainRandCfg, events, track: TrackFn) -> None:
    """Randomize contact materials and plug mass.

    Material terms move from startup to reset and route through
    :class:`~isaaclab_tasks.contrib.deploy.mdp.AdrRigidBodyMaterial`, because the stock term
    caches its PhysX material buckets in its constructor and would ignore a range widened
    by the curriculum.
    """
    material_knobs = (
        ("finger_friction", ("robot_physics_material",)),
        ("mating_friction", ("plug_physics_material", "socket_physics_material")),
    )
    for knob_name, term_names in material_knobs:
        knob: ScalarKnobCfg = getattr(dr, knob_name)
        if not knob.enable:
            continue
        for term_name in term_names:
            term = getattr(events, term_name)
            term.func = mdp.AdrRigidBodyMaterial
            term.mode = "reset"
            term.params["static_friction_range"] = tuple(knob.initial)
            term.params["dynamic_friction_range"] = tuple(knob.initial)
            for field in ("static_friction_range", "dynamic_friction_range"):
                track(f"events.{term_name}.params.{field}", tuple(knob.initial), tuple(knob.final))

    if not dr.plug_mass.enable:
        return
    events.randomize_plug_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("dp_plug", body_names=".*"),
            "mass_distribution_params": tuple(dr.plug_mass.initial),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    track(
        "events.randomize_plug_mass.params.mass_distribution_params",
        tuple(dr.plug_mass.initial),
        tuple(dr.plug_mass.final),
    )


def _apply_reset_state(dr: DomainRandCfg, events, track: TrackFn) -> None:
    """Randomize how the plug is held and where the socket sits."""
    if dr.grasp_pos.enable:
        initial = _symmetric_range(dr.grasp_pos.initial, _AXIS_KEYS)
        events.set_robot_to_grasp_pose.params["pos_randomization_range"] = initial
        track(
            "events.set_robot_to_grasp_pose.params.pos_randomization_range",
            initial,
            _symmetric_range(dr.grasp_pos.final, _AXIS_KEYS),
        )

    if dr.grasp_rot.enable:
        initial = _symmetric_range(dr.grasp_rot.initial, _ROT_KEYS)
        events.set_robot_to_grasp_pose.params["rot_randomization_range"] = initial
        track(
            "events.set_robot_to_grasp_pose.params.rot_randomization_range",
            initial,
            _symmetric_range(dr.grasp_rot.final, _ROT_KEYS),
        )

    if not (dr.socket_pos.enable or dr.socket_rot.enable):
        return
    initial = _pose_range(dr.socket_pos, dr.socket_rot)
    # The stock term caches its range at construction and would ignore the curriculum.
    events.randomize_socket_pose.func = mdp.AdrResetRootStateUniform
    events.randomize_socket_pose.params["pose_range"] = initial
    track(
        "events.randomize_socket_pose.params.pose_range",
        initial,
        _pose_range(
            dr.socket_pos.final if dr.socket_pos.enable else dr.socket_pos.initial,
            dr.socket_rot.final if dr.socket_rot.enable else dr.socket_rot.initial,
        ),
    )


def _apply_disturbances(dr: DomainRandCfg, events, track: TrackFn) -> None:
    """Apply a random wrench to the plug.

    Stands in chiefly for the DisplayPort cable, which hangs off the real plug and exerts a
    time-varying tug that the simulation does not model at all.
    """
    if not (dr.plug_wrench_force.enable or dr.plug_wrench_torque.enable):
        return
    events.plug_wrench = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=tuple(dr.wrench_interval_s),
        params={
            "asset_cfg": SceneEntityCfg("dp_plug", body_names=".*"),
            "force_range": tuple(dr.plug_wrench_force.initial),
            "torque_range": tuple(dr.plug_wrench_torque.initial),
        },
    )
    if dr.plug_wrench_force.enable:
        track(
            "events.plug_wrench.params.force_range",
            tuple(dr.plug_wrench_force.initial),
            tuple(dr.plug_wrench_force.final),
        )
    if dr.plug_wrench_torque.enable:
        track(
            "events.plug_wrench.params.torque_range",
            tuple(dr.plug_wrench_torque.initial),
            tuple(dr.plug_wrench_torque.final),
        )


def _apply_observation_noise(dr: DomainRandCfg, env_cfg, track: TrackFn) -> None:
    """Add correlated (per-episode bias) and uncorrelated (per-step) observation noise."""
    noisy_terms = [(name, term) for name, term in _OBS_TERMS.items() if getattr(dr, name).enable]
    if not noisy_terms:
        return
    env_cfg.observations.policy.enable_corruption = True
    for knob_name, term_name in noisy_terms:
        knob: SignalNoiseKnobCfg = getattr(dr, knob_name)
        obs_term = getattr(env_cfg.observations.policy, term_name)
        obs_term.noise = NoiseModelWithAdditiveBiasCfg(
            noise_cfg=UniformNoiseCfg(n_min=-knob.noise_initial, n_max=knob.noise_initial, operation="add"),
            bias_noise_cfg=UniformNoiseCfg(n_min=-knob.bias_initial, n_max=knob.bias_initial, operation="add"),
            sample_bias_per_component=True,
        )
        address = f"observations.policy.{term_name}.noise"
        for field, initial, final in (
            ("noise_cfg.n_min", -knob.noise_initial, -knob.noise_final),
            ("noise_cfg.n_max", knob.noise_initial, knob.noise_final),
            ("bias_noise_cfg.n_min", -knob.bias_initial, -knob.bias_final),
            ("bias_noise_cfg.n_max", knob.bias_initial, knob.bias_final),
        ):
            track(f"{address}.{field}", initial, final)


def _apply_action_noise(dr: DomainRandCfg, env_cfg, track: TrackFn) -> None:
    """Swap in the OSC action term that models command noise and transport latency."""
    if not (dr.action_noise.enable or dr.action_latency.enable):
        return
    # Carry over every configured field. The subclass is behaviourally identical to the
    # stock term while the noise and latency knobs are zero, including for LEAPP export.
    stock_action = env_cfg.actions.arm_action
    arm_action = mdp.NoisyDelayedOperationalSpaceControllerActionCfg()
    for field, value in stock_action.__dict__.items():
        if field != "class_type":
            setattr(arm_action, field, value)
    env_cfg.actions.arm_action = arm_action

    if dr.action_noise.enable:
        arm_action.action_bias_halfwidth = dr.action_noise.bias_initial
        arm_action.action_noise_halfwidth = dr.action_noise.noise_initial
        track("actions.arm_action.action_bias_halfwidth", dr.action_noise.bias_initial, dr.action_noise.bias_final)
        track("actions.arm_action.action_noise_halfwidth", dr.action_noise.noise_initial, dr.action_noise.noise_final)
    if dr.action_latency.enable:
        arm_action.latency_steps_range = tuple(dr.action_latency.initial)
        arm_action.max_latency_steps = max(arm_action.max_latency_steps, int(round(dr.action_latency.final[1])))
        track(
            "actions.arm_action.latency_steps_range",
            tuple(dr.action_latency.initial),
            tuple(dr.action_latency.final),
        )


def _apply_at_goal_schedule(dr: DomainRandCfg, events, schedule: list) -> None:
    """Select how the existing at-goal plug reset curriculum anneals."""
    params = events.reset_plug_curriculum.params
    if dr.at_goal_schedule == "iteration":
        return
    if dr.at_goal_schedule == "off":
        params["at_goal_prob_final"] = params["at_goal_prob"]
        return
    if dr.at_goal_schedule != "adr":
        raise ValueError(f"Unknown at_goal_schedule '{dr.at_goal_schedule}'. Expected 'iteration', 'adr', or 'off'.")

    # Freeze the iteration-based anneal and let the curriculum drive it instead, so the
    # plug only starts further from the goal once the policy is actually seating it.
    initial_prob = params["at_goal_prob"]
    final_prob = params["at_goal_prob_final"]
    params["at_goal_prob_final"] = initial_prob
    if dr.adr.enable:
        schedule.append(("events.reset_plug_curriculum.params.at_goal_prob", initial_prob, final_prob))
        schedule.append(("events.reset_plug_curriculum.params.at_goal_prob_final", initial_prob, final_prob))


def _build_curriculum(dr: DomainRandCfg, schedule: list) -> dict[str, CurrTerm]:
    """Build the scheduler plus one interpolation term per tracked range."""
    curriculum: dict[str, CurrTerm] = {
        "adr": CurrTerm(
            func=mdp.SuccessDifficultyScheduler,
            params={
                "num_levels": dr.adr.num_levels,
                "success_threshold": dr.adr.success_threshold,
                "min_steps_between": dr.adr.min_steps_between,
                "init_level": dr.adr.init_level,
                "demote": dr.adr.demote,
            },
        )
    }
    for index, (address, initial, final) in enumerate(schedule):
        curriculum[f"adr_{index:02d}_{address.split('.')[-1]}"] = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": address,
                "modify_fn": mdp.interpolate_range_fn,
                "modify_params": {
                    "initial_value": initial,
                    "final_value": final,
                    "difficulty_term_str": "adr",
                },
            },
        )
    return curriculum


def apply_domain_randomization(env_cfg: Rizon4sTaskSpaceDisplayportInsertionEnvCfg) -> None:
    """Apply ``env_cfg.dr`` to the environment configuration in place.

    Does nothing when the master switch is off, so the unrandomized env is untouched.
    Every knob writes its ``initial`` range into the corresponding term; when the ADR
    curriculum is enabled, a matching interpolation term widens that range toward
    ``final`` as the policy succeeds.
    """
    dr: DomainRandCfg = env_cfg.dr
    if not dr.enabled:
        return

    events = env_cfg.events
    # (address, initial, final) for every range the curriculum should widen.
    schedule: list[tuple[str, object, object]] = []

    def track(address: str, initial, final) -> None:
        """Register a range for the curriculum to widen, if it actually widens."""
        if dr.adr.enable and initial != final:
            schedule.append((address, initial, final))

    _apply_controller_gains(dr, events, track)
    _apply_joint_properties(dr, env_cfg, events, track)
    _apply_contact_properties(dr, events, track)
    _apply_reset_state(dr, events, track)
    _apply_disturbances(dr, events, track)
    _apply_observation_noise(dr, env_cfg, track)
    _apply_action_noise(dr, env_cfg, track)
    _apply_at_goal_schedule(dr, events, schedule)

    if dr.adr.enable:
        env_cfg.curriculum = _build_curriculum(dr, schedule)
