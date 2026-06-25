# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import newton
import numpy as np
import pytest
from isaaclab_newton.cloner import replicate as replicate_module
from isaaclab_newton.cloner.replicate import _configure_hydroelastic_sdf_shapes

from pxr import Usd, UsdGeom, UsdPhysics


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


class _FakeMeshBuilder:
    def __init__(self):
        self.body_label = ["/World/Body"]
        self.shape_body = [0, 0, 0]
        self.shape_type = [int(newton.GeoType.MESH)] * 3
        self.shape_flags = [
            int(newton.ShapeFlags.COLLIDE_SHAPES),
            int(newton.ShapeFlags.COLLIDE_SHAPES),
            0,
        ]
        self.shape_label = [
            "/World/Body/SdfCollider",
            "/World/Body/RegularCollider",
            "/World/Body/VisualMesh",
        ]
        self.approximate_meshes_calls = []

    @property
    def shape_count(self):
        return len(self.shape_label)

    def approximate_meshes(self, approximation, shape_indices, keep_visual_shapes):
        self.approximate_meshes_calls.append((approximation, tuple(shape_indices), keep_visual_shapes))


def _make_mesh_collision_stage() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Body")
    for path, approximation in (
        ("/World/Body/SdfCollider", "sdf"),
        ("/World/Body/RegularCollider", "convexHull"),
        ("/World/Body/VisualMesh", "sdf"),
    ):
        prim = UsdGeom.Mesh.Define(stage, path).GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_collision_api.CreateApproximationAttr().Set(approximation)
    return stage


def test_recover_and_simplify_preserves_direct_sdf_mesh_colliders(monkeypatch: pytest.MonkeyPatch):
    builder = _FakeMeshBuilder()
    stage = _make_mesh_collision_stage()
    monkeypatch.setattr(replicate_module, "_enable_intended_mesh_colliders", lambda builder, stage: [])

    replicate_module._recover_and_simplify_source_builders(
        {"source": builder},
        stage,
        authored_bodies=set(),
        simplify_meshes=True,
        preserve_sdf_meshes=True,
    )

    assert builder.approximate_meshes_calls == [("convex_hull", (1,), True)]


def test_recover_and_simplify_hulls_direct_sdf_mesh_colliders_when_not_preserved(
    monkeypatch: pytest.MonkeyPatch,
):
    builder = _FakeMeshBuilder()
    stage = _make_mesh_collision_stage()
    monkeypatch.setattr(replicate_module, "_enable_intended_mesh_colliders", lambda builder, stage: [])

    replicate_module._recover_and_simplify_source_builders(
        {"source": builder},
        stage,
        authored_bodies=set(),
        simplify_meshes=True,
        preserve_sdf_meshes=False,
    )

    assert builder.approximate_meshes_calls == [("convex_hull", (0, 1), True)]
