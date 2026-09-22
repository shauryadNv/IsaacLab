# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Toggleable domain randomization for the Rizon 4S DisplayPort task-space env.

Attached as ``env.dr`` on the task-space env cfg, which the ``-ROS-Inference`` task
inherits. Every knob is off by default, so the unmodified env reproduces the
pre-randomization baseline exactly.

Each knob carries two endpoints: ``initial`` (the range used at ADR level 0, or for
the whole run when the curriculum is off) and ``final`` (the range reached at
``dr.adr.num_levels``). Setting ``initial == final`` gives fixed-range randomization
with no curriculum. Disabling a knob is either ``enable=false`` or collapsing its
range to a constant.

Hydra / CLI examples::

    # Everything on, success-driven ADR widening the ranges
    env.dr.enabled=true env.dr.adr.enable=true

    # Grasp-pose randomization only, fixed range, no curriculum
    env.dr.enabled=true \\
      env.dr.grasp_pos.enable=true env.dr.grasp_pos.initial=[0.002,0.002,0.003]

    # OSC gains over a wider band than the default ADR endpoint
    env.dr.enabled=true env.dr.osc_stiffness.enable=true \\
      env.dr.osc_stiffness.final=[0.4,2.5]

    # Retune the ADR trigger
    env.dr.adr.success_threshold=0.5 env.dr.adr.min_steps_between=6000
"""

from __future__ import annotations

import math

from isaaclab.utils.configclass import configclass


@configclass
class ScalarKnobCfg:
    """A randomization knob whose range is a single ``(low, high)`` pair."""

    enable: bool = False
    """Whether this randomization is applied at all."""

    initial: tuple[float, float] = (0.0, 0.0)
    """Sampling range at ADR level 0, or for the whole run when the curriculum is off."""

    final: tuple[float, float] = (0.0, 0.0)
    """Sampling range at the maximum ADR level."""


@configclass
class AxisKnobCfg:
    """A knob whose range is a symmetric per-axis magnitude, sampled as ``+/- value``."""

    enable: bool = False
    """Whether this randomization is applied at all."""

    initial: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Per-axis half-width at ADR level 0."""

    final: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Per-axis half-width at the maximum ADR level."""


@configclass
class SignalNoiseKnobCfg:
    """Correlated + uncorrelated noise for one observation or action signal.

    Follows the decomposition used by DextrAH-G and ADEPT: a *correlated* component
    sampled once per episode and held (a calibration-style bias), plus an
    *uncorrelated* component resampled every step (sensor jitter). Both are uniform
    and symmetric, expressed as a half-width in the signal's own units.
    """

    enable: bool = False
    """Whether this noise is applied at all."""

    bias_initial: float = 0.0
    """Per-episode bias half-width at ADR level 0. Sampled independently per component."""

    bias_final: float = 0.0
    """Per-episode bias half-width at the maximum ADR level."""

    noise_initial: float = 0.0
    """Per-step noise half-width at ADR level 0."""

    noise_final: float = 0.0
    """Per-step noise half-width at the maximum ADR level."""


@configclass
class AdrCfg:
    """Success-driven automatic domain randomization schedule.

    A single global integer level in ``[0, num_levels]`` advances when the running
    terminal success rate clears :attr:`success_threshold` and at least
    :attr:`min_steps_between` policy steps have passed since the last change. Every
    enabled knob's range is linearly interpolated from its ``initial`` to its
    ``final`` endpoint by ``level / num_levels``.

    Matches the scheme used by ADEPT (50 levels, 0.4 success trigger) and the
    dextrah-unified codebase (same, plus a minimum-steps guard).
    """

    enable: bool = False
    """Whether to advance randomization ranges with training progress."""

    num_levels: int = 50
    """Number of discrete difficulty levels between the ``initial`` and ``final`` ranges."""

    init_level: int = 0
    """Starting level. Set above 0 to resume mid-curriculum or to skip early levels."""

    success_threshold: float = 0.4
    """Terminal success rate that must be exceeded to advance one level."""

    min_steps_between: int = 3000
    """Minimum policy steps between level changes. Prevents runaway advancement."""

    demote: bool = False
    """Whether the level may decrease when success falls below the threshold.

    Off by default: both reference papers advance monotonically.
    """


@configclass
class DomainRandCfg:
    """Master domain-randomization config for the task-space DisplayPort insertion env.

    Note:
        Randomizing the *arm joint* PD gains is deliberately absent. The task-space env
        zeroes arm stiffness and damping so the OSC can drive the joints with pure
        torque, which makes joint-PD randomization a no-op here. The equivalent knobs
        are :attr:`osc_stiffness` / :attr:`osc_damping_ratio` (the gains actually in the
        loop) and :attr:`joint_armature` / :attr:`joint_friction` (which act on the plant
        regardless of the drive).
    """

    enabled: bool = False
    """Master switch. When False no randomization is applied and no curriculum is built."""

    adr: AdrCfg = AdrCfg()
    """Success-driven curriculum over every enabled knob."""

    # ------------------------------------------------------------------
    # Controller
    # ------------------------------------------------------------------

    osc_stiffness: ScalarKnobCfg = ScalarKnobCfg(initial=(1.0, 1.0), final=(0.5, 2.0))
    """Multiplicative scale on the OSC task-space stiffness, sampled per env at reset.

    With ``inertial_dynamics_decoupling=False`` these gains are in N/m and N*m/rad, so a
    scale of 0.5-2.0 spans genuinely softer and stiffer Cartesian behavior. Both
    reference papers randomize controller stiffness over x[0.5, 2].
    """

    osc_damping_ratio: ScalarKnobCfg = ScalarKnobCfg(initial=(1.0, 1.0), final=(0.7, 1.4))
    """Multiplicative scale on the OSC damping ratio, sampled per env at reset.

    Narrower than the stiffness band: the nominal rotational ratio is already near 0.1
    and driving it lower risks an underdamped final approach.
    """

    # ------------------------------------------------------------------
    # Actuation / plant
    # ------------------------------------------------------------------

    joint_armature: ScalarKnobCfg = ScalarKnobCfg(initial=(0.0, 0.0), final=(0.0, 0.01))
    """Absolute arm joint armature [kg*m^2], sampled per env at startup.

    Armature enters the mass matrix, but the OSC law with decoupling off never reads the
    mass matrix, so this creates a true plant/controller mismatch. Applied with
    ``operation="abs"`` because the Flexiv asset's armature baseline is exactly 0.0,
    which makes a multiplicative scale a no-op.
    """

    joint_friction: ScalarKnobCfg = ScalarKnobCfg(initial=(0.0, 0.0), final=(0.0, 0.15))
    """Absolute arm joint friction coefficient [N*m], sampled per env at startup.

    The sim baseline is 0.0 while real harmonic drives have meaningful friction. The
    final endpoint matches the range used on ``ashwinvk/dp_cable_ship_joint_friction_rand``.
    Applied with ``operation="abs"`` for the same reason as :attr:`joint_armature`.
    """

    # ------------------------------------------------------------------
    # Contact
    # ------------------------------------------------------------------

    finger_friction: ScalarKnobCfg = ScalarKnobCfg(initial=(0.75, 0.75), final=(0.4, 1.1))
    """Static/dynamic friction of the gripper finger bodies, sampled per env at startup.

    Governs grasp security. Both reference papers randomize robot friction over
    approximately x[0.5, 1.1] of nominal.
    """

    mating_friction: ScalarKnobCfg = ScalarKnobCfg(initial=(0.001, 0.001), final=(0.001, 0.001))
    """Static/dynamic friction of the plug and socket bodies, sampled per env at startup.

    Off by default, and deliberately so. The env mates the pair at mu = 0.001 with
    ``friction_combine_mode="multiply"``, i.e. near-frictionless on purpose, and the
    dextrah-unified codebase attaches no randomization at all to its tight-tolerance
    assembly parts. Widening this is the largest behavioral change available here;
    treat it as its own experiment rather than part of a default DR bundle.
    """

    plug_mass: ScalarKnobCfg = ScalarKnobCfg(initial=(1.0, 1.0), final=(0.5, 2.0))
    """Multiplicative scale on the plug mass, sampled per env at startup.

    Off by default. The papers use x[0.3, 3] for free objects; the plug is only 30 g and
    is held rigidly, so the effect here is weaker than in a grasping task.
    """

    # ------------------------------------------------------------------
    # Reset state
    # ------------------------------------------------------------------

    grasp_pos: AxisKnobCfg = AxisKnobCfg(initial=(0.0, 0.0, 0.0), final=(0.002, 0.002, 0.003))
    """Per-axis half-width [m] of the plug's translational offset within the gripper.

    The env snaps the plug to the achieved hand pose at reset, so without this the plug
    is held perfectly every episode. Neither reference paper randomizes grasp pose --
    in both, the robot picks the object up itself so grasp variation is emergent. Ours
    is scripted, which makes real-world grasp error entirely unmodeled unless injected
    here.
    """

    grasp_rot: AxisKnobCfg = AxisKnobCfg(
        initial=(0.0, 0.0, 0.0),
        final=(math.radians(1.5), math.radians(1.5), math.radians(1.5)),
    )
    """Per-axis half-width [rad] of roll/pitch/yaw perturbation on the grasp orientation."""

    socket_pos: AxisKnobCfg = AxisKnobCfg(initial=(0.01, 0.01, 0.02), final=(0.025, 0.025, 0.02))
    """Per-axis half-width [m] of socket position randomization at reset.

    ``initial`` reproduces the pre-DR baseline. Keep the final endpoint inside the arm's
    reachable workspace and the perception field of view.
    """

    socket_rot: AxisKnobCfg = AxisKnobCfg(
        initial=(math.radians(2.0),) * 3,
        final=(math.radians(4.0),) * 3,
    )
    """Per-axis half-width [rad] of socket roll/pitch/yaw randomization at reset."""

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    obs_socket_pos: SignalNoiseKnobCfg = SignalNoiseKnobCfg(bias_final=0.005, noise_final=0.0005)
    """Noise on the observed socket keypoint position [m].

    Bias-dominated on purpose: on the real robot the socket pose comes from a one-shot
    perception estimate, so its error is a fixed offset for the whole insertion rather
    than per-frame jitter.
    """

    obs_eef_pos: SignalNoiseKnobCfg = SignalNoiseKnobCfg(bias_final=0.003, noise_final=0.0005)
    """Noise on the observed end-effector (TCP) position [m].

    Bias-dominated: TCP calibration error is constant per setup.
    """

    obs_eef_rot: SignalNoiseKnobCfg = SignalNoiseKnobCfg(bias_final=math.radians(1.0), noise_final=0.0)
    """Noise on the observed end-effector 6D rotation, as a half-width on each component."""

    obs_socket_rot: SignalNoiseKnobCfg = SignalNoiseKnobCfg(bias_final=math.radians(1.0), noise_final=0.0)
    """Noise on the observed socket 6D rotation, as a half-width on each component."""

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    action_noise: SignalNoiseKnobCfg = SignalNoiseKnobCfg(bias_final=0.005, noise_final=0.02)
    """Noise on the policy action before it reaches the controller, in action units.

    Actions are unit-scaled pose deltas (``_ACTION_SCALE`` = 0.025 m/rad per unit), so a
    half-width of 0.02 is roughly 2% of the per-step command. Only DextrAH-G of the two
    reference papers randomizes actions; ADEPT does not.
    """

    action_latency: ScalarKnobCfg = ScalarKnobCfg(initial=(0.0, 0.0), final=(0.0, 2.0))
    """Command latency in whole control steps, sampled per env at reset.

    Neither reference paper models actuation delay. This is motivated by the real
    deployment path instead, where the policy output traverses ROS and the Flexiv RDK
    before reaching the joint controller. At 30 Hz control, 2 steps is ~66 ms.
    """

    # ------------------------------------------------------------------
    # Disturbances
    # ------------------------------------------------------------------

    plug_wrench_force: ScalarKnobCfg = ScalarKnobCfg(initial=(0.0, 0.0), final=(-0.6, 0.6))
    """Per-axis range [N] of a random force on the plug, in the plug frame, resampled on an interval.

    Each of x, y, z is drawn independently from this range, so it must be symmetric for the
    force to point in a random direction; a range like ``(0, 1)`` would always push along the
    plug's +x+y+z diagonal. At ``(-0.6, 0.6)`` the magnitude is ~0.6 N typically and at most
    ~1.04 N.

    Off by default. Stands in chiefly for the DisplayPort cable, which hangs off the real
    plug and exerts a time-varying tug that the sim does not model at all. ~1 N is
    comparable to a real cable pull and well inside the OSC's ~7.5 N per-step authority.
    Both reference papers credit object wrenches with preventing brittle contact
    strategies. Cleared at every episode reset, so each episode's first wrench arrives
    after one resampling interval.
    """

    plug_wrench_torque: ScalarKnobCfg = ScalarKnobCfg(initial=(0.0, 0.0), final=(-0.012, 0.012))
    """Per-axis range [N*m] of a random torque on the plug, in the plug frame.

    Symmetric for the same reason as :attr:`plug_wrench_force`; at most ~0.02 N*m.
    """

    wrench_interval_s: tuple[float, float] = (0.5, 2.0)
    """Resampling interval range [s] for the plug wrench."""

    # ------------------------------------------------------------------
    # Existing at-goal reset curriculum
    # ------------------------------------------------------------------

    at_goal_schedule: str = "iteration"
    """How the existing at-goal plug reset curriculum anneals.

    * ``iteration`` -- anneal on the training-iteration counter (the pre-DR behavior,
      preserved so existing runs stay reproducible).
    * ``adr`` -- anneal on the ADR level instead, so the plug only starts further from
      the goal once the policy is actually succeeding.
    * ``off`` -- hold ``at_goal_prob`` at its initial value.
    """

    def knob_items(self) -> list[tuple[str, object]]:
        """Return ``(name, knob)`` for every knob field, in declaration order."""
        return [
            (name, value)
            for name, value in self.__dict__.items()
            if isinstance(value, (ScalarKnobCfg, AxisKnobCfg, SignalNoiseKnobCfg))
        ]

    def active_knobs(self) -> list[tuple[str, object]]:
        """Return the knobs that are enabled, or all of them when the master switch is off."""
        if not self.enabled:
            return []
        return [(name, knob) for name, knob in self.knob_items() if knob.enable]

    def summary(self) -> str:
        """One-line description of the active configuration, for logs."""
        if not self.enabled:
            return "dr=off"
        active = [name for name, _ in self.active_knobs()]
        if not active:
            return "dr=on(no knobs enabled)"
        adr = f"adr={self.adr.num_levels}L@{self.adr.success_threshold:g}" if self.adr.enable else "adr=off"
        return f"dr=on({adr}, at_goal={self.at_goal_schedule}, knobs={'+'.join(active)})"
