# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import newton
import numpy as np
import pytest
from isaaclab_newton.cloner.replicate import _configure_hydroelastic_sdf_shapes


def test_configure_hydroelastic_sdf_shapes_builds_mesh_sdf_and_sets_flag(monkeypatch: pytest.MonkeyPatch):
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.gap = 0.01
    body = builder.add_body(label="body")
    mesh = newton.Mesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        indices=np.array([0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3], dtype=np.int32),
    )
    shape = builder.add_shape_mesh(body=body, mesh=mesh, scale=(2.0, 1.0, 1.0))
    calls = []

    def build_sdf(self, **kwargs):
        calls.append(kwargs)
        self.sdf = object()
        return self.sdf

    monkeypatch.setattr(newton.Mesh, "build_sdf", build_sdf)

    marked = _configure_hydroelastic_sdf_shapes({"body": builder})

    assert marked == 1
    assert int(builder.shape_flags[shape]) & int(newton.ShapeFlags.HYDROELASTIC)
    assert builder.shape_type[shape] == newton.GeoType.MESH
    assert builder.shape_scale[shape] == (1.0, 1.0, 1.0)
    assert calls == [
        {
            "max_resolution": 64,
            "narrow_band_range": (-0.01, 0.01),
            "margin": 0.01,
        }
    ]
