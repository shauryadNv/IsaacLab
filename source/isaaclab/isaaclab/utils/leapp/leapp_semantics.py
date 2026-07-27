# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LEAPP semantic metadata helpers for raw tensor-producing functions."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

try:
    from leapp import InputKindEnum, OutputKindEnum
except ImportError:

    class _LeappEnumSentinel:
        """Stand-in when leapp is not installed.

        Any attribute access returns ``None`` so that
        ``@leapp_tensor_semantics(kind=InputKindEnum.BODY_POSE)``
        silently stores ``kind=None`` instead of crashing at import time.
        The real enum values are only needed at export time, when leapp
        *is* guaranteed to be available.
        """

        def __getattr__(self, name: str):
            return None

    InputKindEnum = _LeappEnumSentinel()  # type: ignore[assignment,misc]
    OutputKindEnum = _LeappEnumSentinel()  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class LeappTensorSemantics:
    """Semantic metadata attached directly to a raw tensor-producing function."""

    kind: Any = None
    element_names: list[str] | list[list[str]] | None = None
    element_names_resolver: Callable | None = None
    const: bool = False


XYZ_ELEMENT_NAMES: list[str] = ["x", "y", "z"]
QUAT_XYZW_ELEMENT_NAMES: list[str] = ["qx", "qy", "qz", "qw"]
POSE7_ELEMENT_NAMES: list[str] = ["x", "y", "z", "qx", "qy", "qz", "qw"]
POSE6_ELEMENT_NAMES: list[str] = ["x", "y", "z", "angular_x", "angular_y", "angular_z"]
WRENCH6_ELEMENT_NAMES: list[str] = ["fx", "fy", "fz", "tx", "ty", "tz"]
TWIST3_ELEMENT_NAMES: list[str] = ["lin_vel_x", "lin_vel_y", "ang_vel_z"]
TWIST6_ELEMENT_NAMES: list[str] = [
    "lin_vel_x",
    "lin_vel_y",
    "lin_vel_z",
    "ang_vel_x",
    "ang_vel_y",
    "ang_vel_z",
]

_COMMAND_BODY_VELOCITY_KIND = "command/body/velocity"
_TWIST_ELEMENT_NAME_ALIASES = {
    ("lin_x", "lin_y", "ang_z"): TWIST3_ELEMENT_NAMES,
    ("lin_x", "lin_y", "lin_z", "ang_x", "ang_y", "ang_z"): TWIST6_ELEMENT_NAMES,
    ("linear_x", "linear_y", "angular_z"): TWIST3_ELEMENT_NAMES,
    ("linear_x", "linear_y", "linear_z", "angular_x", "angular_y", "angular_z"): TWIST6_ELEMENT_NAMES,
}


def select_element_names(names: list[str] | None, indices: Any = None) -> list[str] | None:
    """Select element names using optional runtime indices."""
    if names is None:
        return None
    if indices is None or indices == slice(None):
        return list(names)
    if isinstance(indices, slice):
        return list(names[indices])
    with suppress(AttributeError):
        indices = indices.tolist()
    if isinstance(indices, (list, tuple)):
        return [names[int(index)] for index in indices]
    if isinstance(indices, int):
        return [names[indices]]
    return None


def canonicalize_command_element_names(
    kind: Any,
    element_names: list[str] | list[list[str]] | None,
    ref: Any | None = None,
) -> list[str] | list[list[str]] | None:
    """Return Deploy-compatible element names for command tensors.

    Isaac ROS Deploy's Twist/TwistStamped converters publish body velocity
    elements as ``lin_vel_*`` and ``ang_vel_*``.  Older export configs used
    shorter names such as ``lin_x`` / ``ang_x``.  Canonicalize only the
    command-body-velocity case so tensor ordering stays unchanged while the
    exported metadata matches the runtime converter ``TensorSpec``.
    """
    if getattr(kind, "value", kind) != _COMMAND_BODY_VELOCITY_KIND:
        return element_names

    width = None
    with suppress(AttributeError, IndexError, TypeError):
        width = int(ref.shape[-1])
    if width is not None and width not in (3, 6):
        raise ValueError(f"LEAPP command/body/velocity input must be 3D or 6D, but tensor width is {width}.")

    if element_names is None:
        if width == 3:
            return TWIST3_ELEMENT_NAMES
        if width == 6:
            return TWIST6_ELEMENT_NAMES
        return None

    # Command tensors are flat.  Nested names cannot match the current Deploy
    # Twist converter TensorSpec.
    if any(isinstance(name, (list, tuple)) for name in element_names):
        raise ValueError("LEAPP command/body/velocity element names must be a flat list.")

    names_tuple = tuple(element_names)
    canonical_names = _TWIST_ELEMENT_NAME_ALIASES.get(names_tuple, list(element_names))
    if len(canonical_names) == 3:
        expected_names = TWIST3_ELEMENT_NAMES
    elif len(canonical_names) == 6:
        expected_names = TWIST6_ELEMENT_NAMES
    else:
        raise ValueError(
            "LEAPP command/body/velocity input must have 3 or 6 element names, "
            f"but got {len(canonical_names)}."
        )

    if width is not None and len(canonical_names) != width:
        raise ValueError(
            f"LEAPP command/body/velocity input has {len(canonical_names)} element names, "
            f"but tensor width is {width}."
        )
    if canonical_names != expected_names:
        raise ValueError(
            "LEAPP command/body/velocity element names must match Isaac ROS Deploy Twist converter names. "
            f"Got {list(element_names)}; expected {expected_names}."
        )

    return list(canonical_names)


def leapp_tensor_semantics(
    *,
    kind: Any = None,
    element_names: list[str] | list[list[str]] | None = None,
    element_names_resolver: Callable | None = None,
    const: bool = False,
) -> Callable:
    """Attach LEAPP semantic metadata to a raw tensor-producing function."""

    semantics = LeappTensorSemantics(
        kind=kind,
        element_names=element_names,
        element_names_resolver=element_names_resolver,
        const=const,
    )

    def _apply(func: Callable) -> Callable:
        func._leapp_semantics = semantics
        return func

    return _apply


def resolve_leapp_element_names(semantics: LeappTensorSemantics | None, data_self) -> list | None:
    """Resolve element names from attached semantics and a tensor-producing object."""
    if semantics is None:
        return None
    if semantics.element_names is not None:
        return semantics.element_names
    if semantics.element_names_resolver is not None:
        return semantics.element_names_resolver(data_self)
    return None


# ── Predefined element-name resolvers ─────────────────────────────


def joint_names_resolver(data_self) -> list[str] | None:
    """Resolve joint element names from the data object at trace time."""
    return select_element_names(
        getattr(data_self, "joint_names", getattr(data_self, "_joint_names", None)),
        getattr(data_self, "_joint_ids", None),
    )


def body_names_resolver(data_self) -> list[str] | None:
    """Resolve body element names from the data object at trace time."""
    return select_element_names(
        getattr(data_self, "body_names", getattr(data_self, "_body_names", None)),
        getattr(data_self, "_body_ids", None),
    )


def _compound_resolver(outer_fn: Callable, inner_names: list[str]) -> Callable:
    """Build a 2D resolver: ``[outer_names, inner_constant_names]``."""

    def resolver(data_self) -> list | None:
        outer = outer_fn(data_self)
        return [outer, inner_names] if outer else None

    return resolver


def _target_frame_names(data_self) -> list[str] | None:
    names = getattr(data_self, "target_frame_names", None)
    return list(names) if names is not None else None


body_xyz_resolver = _compound_resolver(body_names_resolver, XYZ_ELEMENT_NAMES)
body_pose_resolver = _compound_resolver(body_names_resolver, POSE7_ELEMENT_NAMES)
body_pose6_resolver = _compound_resolver(body_names_resolver, POSE6_ELEMENT_NAMES)
body_quat_resolver = _compound_resolver(body_names_resolver, QUAT_XYZW_ELEMENT_NAMES)
body_wrench_resolver = _compound_resolver(body_names_resolver, WRENCH6_ELEMENT_NAMES)
target_frame_xyz_resolver = _compound_resolver(_target_frame_names, XYZ_ELEMENT_NAMES)
target_frame_quat_resolver = _compound_resolver(_target_frame_names, QUAT_XYZW_ELEMENT_NAMES)
target_frame_pose_resolver = _compound_resolver(_target_frame_names, POSE7_ELEMENT_NAMES)
