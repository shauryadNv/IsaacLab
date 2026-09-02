# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for DisplayPort task-space LEAPP observation contracts."""

import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXPORT_SCRIPT = (
    _REPO_ROOT / "scripts" / "reinforcement_learning" / "leapp" / "rsl_rl" / "export_displayport_insertion.py"
)
_EXPORT_MODULE_NAME = "_isaaclab_displayport_task_space_export"


def _load_export_module():
    """Load the task-specific exporter without starting Isaac Sim."""
    module = sys.modules.get(_EXPORT_MODULE_NAME)
    if module is not None:
        return module

    spec = importlib.util.spec_from_file_location(_EXPORT_MODULE_NAME, _EXPORT_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {_EXPORT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_EXPORT_MODULE_NAME] = module
    original_export = sys.modules.pop("export", None)
    try:
        spec.loader.exec_module(module)
    finally:
        if original_export is None:
            sys.modules.pop("export", None)
        else:
            sys.modules["export"] = original_export
    return module


def test_task_space_input_spec_defaults_to_physx_contract():
    """Test configs without order metadata retain the EEF-first contract."""
    export_module = _load_export_module()

    resolved = export_module.resolve_task_space_input_spec(SimpleNamespace())

    assert resolved is export_module._TASK_SPACE_INPUT_SPEC
    assert [entry[0] for entry in resolved] == [
        "eef_pos",
        "eef_rot_6d",
        "socket_kp_pos",
        "socket_kp_rot_6d",
    ]
    assert (
        inspect.signature(export_module.split_and_annotate_task_space_obs).parameters["input_spec"].default
        is export_module._TASK_SPACE_INPUT_SPEC
    )


def test_task_space_input_spec_uses_newton_observation_order():
    """Test Newton metadata produces canonical Deploy ports in trained actor order."""
    export_module = _load_export_module()
    obs_order = ["socket_pos", "tool_pos", "tool_rot_6d", "socket_rot_6d"]

    resolved = export_module.resolve_task_space_input_spec(SimpleNamespace(task_space_obs_order=obs_order))

    assert [entry[0] for entry in resolved] == [
        "socket_kp_pos",
        "eef_pos",
        "eef_rot_6d",
        "socket_kp_rot_6d",
    ]
    assert [(entry[1].start, entry[1].stop) for entry in resolved] == [(0, 3), (3, 6), (6, 12), (12, 18)]
    assert [entry[3] for entry in resolved] == [
        "socket_kp_pose_pos",
        "eef_pose_pos",
        "eef_pose_rot6d",
        "socket_kp_pose_rot6d",
    ]


@pytest.mark.parametrize(
    ("obs_order", "message"),
    [
        pytest.param(["socket_pos", "unknown", "tool_rot_6d", "socket_rot_6d"], "Unknown", id="unknown"),
        pytest.param(["socket_pos", "tool_pos", "tool_rot_6d", "tool_rot_6d"], "Duplicate", id="duplicate"),
        pytest.param(["socket_pos", "tool_pos"], "exactly 18", id="wrong-width"),
        pytest.param(
            ["socket_pos", "socket_kp_pos", "tool_rot_6d", "socket_rot_6d"],
            "Duplicate Deploy input",
            id="aliased-input",
        ),
    ],
)
def test_task_space_input_spec_rejects_invalid_metadata(obs_order, message):
    """Test invalid observation ABI metadata fails before policy export."""
    export_module = _load_export_module()

    with pytest.raises(ValueError, match=message):
        export_module.resolve_task_space_input_spec(SimpleNamespace(task_space_obs_order=obs_order))


def test_task_space_action_matches_runner_clip_then_osc_scale():
    """Deployment must reproduce the model-999 raw-action transform."""
    export_module = _load_export_module()
    actions = torch.tensor([[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]])
    scale = torch.full((1, 6), 0.025)

    processed = export_module.process_task_space_action(actions, scale, clip_actions=1.0)

    torch.testing.assert_close(
        processed,
        torch.tensor([[-0.025, -0.025, -0.0125, 0.0125, 0.025, 0.025]]),
    )


def test_task_space_action_scale_expands_scalar_and_per_axis_values(monkeypatch):
    """Test OSC scales accept scalar and three-axis configuration values."""
    export_module = _load_export_module()
    monkeypatch.setattr(export_module._export, "torch", torch)
    env_cfg = SimpleNamespace(
        actions=SimpleNamespace(
            arm_action=SimpleNamespace(
                position_scale=[0.025, 0.025, 0.01],
                orientation_scale=0.025,
            )
        )
    )

    scale = export_module.task_space_action_scale(env_cfg, "cpu", torch.float32)

    torch.testing.assert_close(scale, torch.tensor([[0.025, 0.025, 0.01, 0.025, 0.025, 0.025]]))


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        pytest.param("position_scale", [0.025, 0.025], id="position-two-axis"),
        pytest.param("orientation_scale", [0.025, 0.025, 0.025, 0.025], id="orientation-four-axis"),
    ],
)
def test_task_space_action_scale_rejects_invalid_axis_count(attribute, value, monkeypatch):
    """Test OSC scales reject sequences that are neither scalar nor three-axis."""
    export_module = _load_export_module()
    monkeypatch.setattr(export_module._export, "torch", torch)
    action_cfg = SimpleNamespace(position_scale=0.025, orientation_scale=0.025)
    setattr(action_cfg, attribute, value)
    env_cfg = SimpleNamespace(actions=SimpleNamespace(arm_action=action_cfg))

    with pytest.raises(ValueError, match=f"{attribute} must contain either one or three values"):
        export_module.task_space_action_scale(env_cfg, "cpu", torch.float32)


def test_task_space_action_scale_rejects_non_finite_values(monkeypatch):
    """Test OSC scales reject non-finite deployment transforms."""
    export_module = _load_export_module()
    monkeypatch.setattr(export_module._export, "torch", torch)
    env_cfg = SimpleNamespace(
        actions=SimpleNamespace(
            arm_action=SimpleNamespace(position_scale=[0.025, float("nan"), 0.01], orientation_scale=0.025)
        )
    )

    with pytest.raises(ValueError, match="position_scale must contain only finite values"):
        export_module.task_space_action_scale(env_cfg, "cpu", torch.float32)


def test_task_space_action_can_leave_runner_action_unclipped():
    """A disabled runner clip must not add an exporter-only clamp."""
    export_module = _load_export_module()
    actions = torch.tensor([[-2.0, 2.0]])

    processed = export_module.process_task_space_action(actions, torch.full((1, 2), 0.025), clip_actions=None)

    torch.testing.assert_close(processed, torch.tensor([[-0.05, 0.05]]))
