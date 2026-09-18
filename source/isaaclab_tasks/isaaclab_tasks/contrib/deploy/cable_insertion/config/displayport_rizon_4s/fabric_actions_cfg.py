# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action configuration for the DisplayPort geometric-fabric experiment."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils.configclass import configclass

from .joint_fabric_controller import JointFabricControllerCfg


@configclass
class DisplayportFabricActionCfg(ActionTermCfg):
    """Configure a stateful artificial-joint fabric and explicit joint tracker."""

    class_type: type | str = (
        "isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.fabric_actions:"
        "DisplayportFabricAction"
    )

    joint_names: list[str] = MISSING
    """Seven arm joint names in kinematic-chain order."""

    body_name: str = "flange"
    """Robot body carrying the tool-center point."""

    socket_asset_name: str = "dp_socket"
    """Scene asset defining the socket frame."""

    socket_insertion_offset: tuple[float, float, float] = MISSING
    """Insertion-axis origin relative to the socket frame [m]."""

    socket_tcp_goal_offset: tuple[float, float, float] = MISSING
    """Seated TCP position relative to the socket insertion frame [m]."""

    tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.1925)
    """Tool-center-point translation relative to the flange frame [m]."""

    tracker_actuator_names: tuple[str, ...] = ("shoulder", "elbow", "wrist")
    """Explicit IdealPD actuator groups covering the seven arm joints."""

    fabric_cfg: JointFabricControllerCfg = JointFabricControllerCfg()
    """Artificial-state geometric-fabric configuration."""

    use_inertial_feedforward: bool = True
    r"""Add :math:`M(q)\ddot q_f` to the tracker feed-forward effort."""

    use_gravity_compensation: bool = True
    """Add the articulation gravity-compensation effort exactly once."""

    reset_velocity_from_robot: bool = False
    """Initialize artificial velocity from measured velocity instead of zero."""

    expected_physics_dt: float = 1.0 / 240.0
    """Required PhysX simulation timestep [s]."""

    expected_policy_decimation: int = 8
    """Required number of physics steps per 30 Hz policy step."""

    parity_tracking_tolerance: float = 1.0e-4
    """Maximum tracking error [rad] allowed by the kinematics parity validator."""

    parity_position_tolerance: float = 5.0e-4
    """Maximum TCP position disagreement [m] accepted by parity validation."""

    parity_rotation_tolerance: float = 2.0e-3
    """Maximum TCP rotation-matrix disagreement accepted by parity validation."""

    parity_jacobian_tolerance: float = 2.0e-3
    """Maximum TCP Jacobian disagreement accepted by parity validation."""
