# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("leapp")

from isaaclab.envs import mdp
from isaaclab.test.mock_interfaces.assets.mock_articulation import MockArticulationData
from isaaclab.utils import math as math_utils
from isaaclab.utils.leapp import utils as leapp_utils
from isaaclab.utils.leapp.export_annotator import ExportPatcher
from isaaclab.utils.leapp.leapp_semantics import InputKindEnum, TWIST6_ELEMENT_NAMES
from isaaclab.utils.leapp.proxy import _DataProxy, _EnvProxy


class _TestScene(dict):
    """Minimal scene mapping for LEAPP proxy tests."""

    sensors = {}


def _make_articulation_data() -> tuple[MockArticulationData, torch.Tensor]:
    """Create mock articulation data with a non-identity root orientation."""
    data = MockArticulationData(num_instances=2, num_joints=0, num_bodies=1, device="cpu")
    root_pose_w = torch.zeros(2, 7, dtype=torch.float32)
    root_pose_w[:, 6] = 1.0
    root_pose_w[1, 3] = math.sin(math.pi / 4.0)
    root_pose_w[1, 6] = math.cos(math.pi / 4.0)
    data.set_root_link_pose_w(root_pose_w)
    return data, root_pose_w


def _capture_leapp_inputs(monkeypatch: pytest.MonkeyPatch) -> list:
    """Capture LEAPP input annotations while returning their tensor references."""
    annotated_inputs = []

    def _record_input_tensor(task_name, semantics):
        annotated_inputs.append((task_name, semantics))
        return semantics.ref

    monkeypatch.setattr(leapp_utils.annotate, "input_tensors", _record_input_tensor)
    return annotated_inputs


def test_direct_projected_gravity_b_read_preserves_vector3d_input(monkeypatch: pytest.MonkeyPatch):
    """Test direct data proxy reads keep projected gravity as its own semantic input."""
    annotated_inputs = _capture_leapp_inputs(monkeypatch)
    data, _ = _make_articulation_data()

    proxy = _DataProxy(
        data,
        entity_name="robot",
        task_name="Isaac-Velocity-Flat-G1",
        property_resolution_cache={},
        cache={},
        input_name_resolver=lambda property_name: f"robot_{property_name}",
    )

    assert proxy.projected_gravity_b.torch.shape == (2, 3)

    assert len(annotated_inputs) == 1
    task_name, semantics = annotated_inputs[0]
    assert task_name == "Isaac-Velocity-Flat-G1"
    assert semantics.name == "robot_projected_gravity_b"
    assert semantics.kind == InputKindEnum.VECTOR3D
    assert semantics.extra == {"isaaclab_connection": "state:robot:projected_gravity_b"}


def test_projected_gravity_observation_exports_root_quat_w_input(monkeypatch: pytest.MonkeyPatch):
    """Test the projected-gravity observation is export-lowered through root quaternion."""
    annotated_inputs = _capture_leapp_inputs(monkeypatch)
    data, root_pose_w = _make_articulation_data()
    scene = _TestScene({"robot": SimpleNamespace(data=data)})
    env = SimpleNamespace(scene=scene)
    proxy_env = _EnvProxy(env, "Isaac-Velocity-Flat-G1", {}, {})

    term_cfg = SimpleNamespace(func=mdp.projected_gravity, noise="noise")
    obs_manager = SimpleNamespace(_group_obs_term_cfgs={"policy": [term_cfg]}, compute=lambda *args, **kwargs: None)
    patcher = ExportPatcher(export_method="onnx-dynamo", required_obs_groups={"policy"})
    patcher.task_name = "Isaac-Velocity-Flat-G1"
    patcher._patch_observation_manager(obs_manager, proxy_env)

    projected_gravity_b = term_cfg.func(env)

    expected = math_utils.quat_apply_inverse(
        root_pose_w[:, 3:7],
        torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32).expand(2, 3),
    )
    assert torch.allclose(projected_gravity_b, expected)
    assert term_cfg.noise is None

    assert len(annotated_inputs) == 1
    task_name, semantics = annotated_inputs[0]
    assert task_name == "Isaac-Velocity-Flat-G1"
    assert semantics.name == "robot_root_quat_w"
    assert semantics.kind == InputKindEnum.BODY_ROTATION
    assert semantics.extra == {"isaaclab_connection": "state:robot:root_quat_w"}


def test_generated_body_velocity_command_exports_deploy_twist_names(monkeypatch: pytest.MonkeyPatch):
    """Test legacy body-velocity command names are exported with Deploy Twist names."""
    annotated_inputs = _capture_leapp_inputs(monkeypatch)
    command_tensor = torch.zeros(2, 6, dtype=torch.float32)

    def generated_commands(env, command_name=None, **kwargs):
        return command_tensor

    command_cfg = SimpleNamespace(
        cmd_kind="command/body/velocity",
        element_names=["lin_x", "lin_y", "lin_z", "ang_x", "ang_y", "ang_z"],
    )
    command_manager = SimpleNamespace(get_term=lambda name: SimpleNamespace(cfg=command_cfg))
    env = SimpleNamespace(command_manager=command_manager)
    term_cfg = SimpleNamespace(params={"command_name": "target_twist"})

    patcher = ExportPatcher(export_method="onnx-dynamo")
    patcher.task_name = "Isaac-Test-Task"

    result = patcher._wrap_generated_commands(generated_commands, term_cfg)(env)

    assert result is command_tensor
    assert len(annotated_inputs) == 1
    task_name, semantics = annotated_inputs[0]
    assert task_name == "Isaac-Test-Task"
    assert semantics.name == "target_twist"
    assert semantics.kind == "command/body/velocity"
    assert semantics.element_names == [TWIST6_ELEMENT_NAMES]
    assert semantics.extra == {"isaaclab_connection": "command:target_twist"}


def test_generated_body_velocity_command_rejects_non_deploy_twist_names(monkeypatch: pytest.MonkeyPatch):
    """Test unknown body-velocity command names fail before exporting unusable YAML."""
    _capture_leapp_inputs(monkeypatch)
    command_tensor = torch.zeros(2, 6, dtype=torch.float32)

    def generated_commands(env, command_name=None, **kwargs):
        return command_tensor

    command_cfg = SimpleNamespace(
        cmd_kind="command/body/velocity",
        element_names=["vx", "vy", "vz", "wx", "wy", "wz"],
    )
    command_manager = SimpleNamespace(get_term=lambda name: SimpleNamespace(cfg=command_cfg))
    env = SimpleNamespace(command_manager=command_manager)
    term_cfg = SimpleNamespace(params={"command_name": "target_twist"})

    patcher = ExportPatcher(export_method="onnx-dynamo")
    patcher.task_name = "Isaac-Test-Task"

    with pytest.raises(ValueError, match="command/body/velocity element names"):
        patcher._wrap_generated_commands(generated_commands, term_cfg)(env)
