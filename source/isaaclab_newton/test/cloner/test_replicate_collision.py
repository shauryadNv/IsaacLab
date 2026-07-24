# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import newton
import numpy as np
import pytest
from isaaclab_newton.cloner import replicate as replicate_module
from isaaclab_newton.cloner.replicate import (
    _configure_hydroelastic_sdf_shapes,
    _configure_sdf_shapes,
    _configure_shape_collision_filter_pairs,
)
from isaaclab_newton.physics import HydroelasticSDFCfg, NewtonCollisionPipelineCfg

from pxr import Usd, UsdGeom, UsdPhysics


@pytest.mark.parametrize(("configured_resolution", "expected_resolution"), [(None, 64), (128, 128)])
def test_configure_hydroelastic_sdf_shapes_builds_mesh_sdf_and_sets_flag(
    monkeypatch: pytest.MonkeyPatch,
    configured_resolution: int | None,
    expected_resolution: int,
):
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

    if configured_resolution is None:
        marked = _configure_hydroelastic_sdf_shapes({"body": builder})
    else:
        marked = _configure_hydroelastic_sdf_shapes({"body": builder}, max_resolution=configured_resolution)

    assert marked == 1
    assert int(builder.shape_flags[shape]) & int(newton.ShapeFlags.HYDROELASTIC)
    assert builder.shape_type[shape] == newton.GeoType.MESH
    assert builder.shape_scale[shape] == (1.0, 1.0, 1.0)
    assert calls == [
        {
            "max_resolution": expected_resolution,
            "narrow_band_range": (-0.01, 0.01),
            "margin": 0.01,
        }
    ]


def test_configure_sdf_shapes_builds_hard_sdf_without_hydroelastic_flag(monkeypatch: pytest.MonkeyPatch):
    """Hard SDF cooking should not opt shapes into hydroelastic contacts."""
    builder = newton.ModelBuilder()
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
    shape = builder.add_shape_mesh(body=body, mesh=mesh)

    monkeypatch.setattr(newton.Mesh, "build_sdf", lambda self, **kwargs: setattr(self, "sdf", object()))

    configured = _configure_sdf_shapes({"body": builder}, max_resolution=128)

    assert configured == 1
    assert mesh.sdf is not None
    assert not int(builder.shape_flags[shape]) & int(newton.ShapeFlags.HYDROELASTIC)


def test_configure_sdf_shapes_only_configures_matching_shape_paths(monkeypatch: pytest.MonkeyPatch):
    builder = newton.ModelBuilder()
    body = builder.add_body(label="body")
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3], dtype=np.int32)
    selected_mesh = newton.Mesh(vertices=vertices, indices=indices)
    skipped_mesh = newton.Mesh(vertices=vertices, indices=indices)
    selected = builder.add_shape_mesh(body=body, mesh=selected_mesh)
    skipped = builder.add_shape_mesh(body=body, mesh=skipped_mesh)
    builder.shape_label[selected] = "/World/DisplayPortPlug/colliders/sdf_connector"
    builder.shape_label[skipped] = "/World/Robot/left_finger/collisions"

    monkeypatch.setattr(newton.Mesh, "build_sdf", lambda self, **kwargs: setattr(self, "sdf", object()))

    configured = _configure_sdf_shapes(
        {"body": builder},
        max_resolution=128,
        enable_hydroelastic=True,
        shape_path_exprs=(r".*/DisplayPortPlug/colliders/sdf_.*",),
    )

    assert configured == 1
    assert selected_mesh.sdf is not None
    assert skipped_mesh.sdf is None
    assert int(builder.shape_flags[selected]) & int(newton.ShapeFlags.HYDROELASTIC)
    assert not int(builder.shape_flags[skipped]) & int(newton.ShapeFlags.HYDROELASTIC)


def test_hydroelastic_sdf_resolution_is_not_forwarded_to_runtime_pipeline():
    cfg = NewtonCollisionPipelineCfg(sdf_hydroelastic_config=HydroelasticSDFCfg(sdf_max_resolution=128))

    pipeline_args = cfg.to_pipeline_args()

    runtime_cfg = pipeline_args["sdf_hydroelastic_config"]
    assert not hasattr(runtime_cfg, "sdf_max_resolution")
    assert runtime_cfg.reduce_contacts is True


def test_hard_sdf_resolution_is_not_forwarded_to_runtime_pipeline():
    cfg = NewtonCollisionPipelineCfg(mesh_sdf_max_resolution=128)

    pipeline_args = cfg.to_pipeline_args()

    assert "mesh_sdf_max_resolution" not in pipeline_args
    assert "sdf_hydroelastic_config" not in pipeline_args


def test_sdf_builder_options_are_not_forwarded_to_runtime_pipeline():
    cfg = NewtonCollisionPipelineCfg(
        mesh_sdf_shape_path_exprs=(r".*/colliders/sdf_.*",),
        preserve_concave_shape_path_exprs=(),
    )

    pipeline_args = cfg.to_pipeline_args()

    assert "mesh_sdf_shape_path_exprs" not in pipeline_args
    assert "preserve_concave_shape_path_exprs" not in pipeline_args


def test_shape_collision_filter_options_are_not_forwarded_to_runtime_pipeline():
    cfg = NewtonCollisionPipelineCfg(
        shape_collision_filter_pair_path_exprs=((r".*/Robot/link.*", r".*/DisplayPort.*"),),
    )

    pipeline_args = cfg.to_pipeline_args()

    assert "shape_collision_filter_pair_path_exprs" not in pipeline_args


def test_configure_shape_collision_filter_pairs_filters_matching_same_world_shapes():
    builder = newton.ModelBuilder()
    builder.begin_world()
    robot_body = builder.add_body(label="/World/envs/env_0/Robot/link1")
    plug_body = builder.add_body(label="/World/envs/env_0/DisplayPortPlug")
    finger_body = builder.add_body(label="/World/envs/env_0/Robot/left_finger_tip")
    robot_shape = builder.add_shape_box(body=robot_body)
    plug_shape = builder.add_shape_box(body=plug_body)
    finger_shape = builder.add_shape_box(body=finger_body)
    builder.end_world()
    builder.shape_label[robot_shape] = "/World/envs/env_0/Robot/link1/collisions"
    builder.shape_label[plug_shape] = "/World/envs/env_0/DisplayPortPlug/colliders/sdf_connector"
    builder.shape_label[finger_shape] = "/World/envs/env_0/Robot/left_finger_tip/collisions"

    added = _configure_shape_collision_filter_pairs(
        builder,
        (
            (
                r".*/Robot/(?!left_finger_tip/).*/?collisions",
                r".*/DisplayPortPlug/.*",
            ),
        ),
    )

    assert added == 1
    assert builder.shape_collision_filter_pairs == [(robot_shape, plug_shape)]


def test_configure_shape_collision_filter_pairs_skips_cross_world_pairs_and_duplicates():
    builder = newton.ModelBuilder()
    for env_id in range(2):
        builder.begin_world()
        robot_body = builder.add_body(label=f"/World/envs/env_{env_id}/Robot/link1")
        plug_body = builder.add_body(label=f"/World/envs/env_{env_id}/DisplayPortPlug")
        robot_shape = builder.add_shape_box(body=robot_body)
        plug_shape = builder.add_shape_box(body=plug_body)
        builder.shape_label[robot_shape] = f"/World/envs/env_{env_id}/Robot/link1/collisions"
        builder.shape_label[plug_shape] = f"/World/envs/env_{env_id}/DisplayPortPlug/collider"
        builder.end_world()

    expr_pairs = ((r".*/Robot/link1/collisions", r".*/DisplayPortPlug/collider"),)

    assert _configure_shape_collision_filter_pairs(builder, expr_pairs) == 2
    assert _configure_shape_collision_filter_pairs(builder, expr_pairs) == 0
    assert builder.shape_collision_filter_pairs == [(0, 1), (2, 3)]


@pytest.mark.parametrize("max_resolution", [0, 63])
def test_configure_hydroelastic_sdf_shapes_rejects_invalid_resolution(max_resolution: int):
    with pytest.raises(ValueError, match="positive and divisible by 8"):
        _configure_hydroelastic_sdf_shapes({}, max_resolution=max_resolution)


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


def test_recover_and_simplify_preserves_matching_concave_meshes(monkeypatch: pytest.MonkeyPatch):
    builder = _FakeMeshBuilder()
    builder.body_label[0] = "/World/Robot/left_finger"
    stage = _make_mesh_collision_stage()
    monkeypatch.setattr(replicate_module, "_enable_intended_mesh_colliders", lambda builder, stage: [])

    replicate_module._recover_and_simplify_source_builders(
        {"source": builder},
        stage,
        authored_bodies=set(),
        simplify_meshes=True,
        preserve_sdf_meshes=False,
        preserve_concave_shape_path_exprs=(r"(?i).*finger.*",),
    )

    assert builder.approximate_meshes_calls == []


def test_recover_and_simplify_hulls_meshes_when_preserve_patterns_are_empty(
    monkeypatch: pytest.MonkeyPatch,
):
    builder = _FakeMeshBuilder()
    builder.body_label[0] = "/World/Robot/left_finger"
    stage = _make_mesh_collision_stage()
    monkeypatch.setattr(replicate_module, "_enable_intended_mesh_colliders", lambda builder, stage: [])

    replicate_module._recover_and_simplify_source_builders(
        {"source": builder},
        stage,
        authored_bodies=set(),
        simplify_meshes=True,
        preserve_sdf_meshes=False,
        preserve_concave_shape_path_exprs=(),
    )

    assert builder.approximate_meshes_calls == [("convex_hull", (0, 1), True)]
