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
    """Test Newton metadata produces contiguous socket/flange actor slices."""
    export_module = _load_export_module()
    obs_order = ["socket_pos", "tool_pos", "tool_rot_6d", "socket_rot_6d"]

    resolved = export_module.resolve_task_space_input_spec(SimpleNamespace(task_space_obs_order=obs_order))

    assert [entry[0] for entry in resolved] == obs_order
    assert [(entry[1].start, entry[1].stop) for entry in resolved] == [(0, 3), (3, 6), (6, 12), (12, 18)]


@pytest.mark.parametrize(
    ("obs_order", "message"),
    [
        pytest.param(["socket_pos", "unknown", "tool_rot_6d", "socket_rot_6d"], "Unknown", id="unknown"),
        pytest.param(["socket_pos", "tool_pos", "tool_rot_6d", "tool_rot_6d"], "Duplicate", id="duplicate"),
        pytest.param(["socket_pos", "tool_pos"], "exactly 18", id="wrong-width"),
    ],
)
def test_task_space_input_spec_rejects_invalid_metadata(obs_order, message):
    """Test invalid observation ABI metadata fails before policy export."""
    export_module = _load_export_module()

    with pytest.raises(ValueError, match=message):
        export_module.resolve_task_space_input_spec(SimpleNamespace(task_space_obs_order=obs_order))
