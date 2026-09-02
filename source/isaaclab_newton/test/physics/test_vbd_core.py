# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the core Newton VBD integration."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
from isaaclab_newton.physics import NewtonCfg, NewtonManager, NewtonSoftContactCfg
from newton.solvers import SolverVBD

from isaaclab.physics import PhysicsManager


def test_vbd_rigid_tuning_fields_are_forwarded_to_newton():
    """Public rigid tuning fields should reach Newton's VBD constructor."""
    physics = importlib.import_module("isaaclab_newton.physics")
    solver_cfg = physics.VBDSolverCfg(
        friction_epsilon=0.02,
        rigid_avbd_alpha=0.9,
        rigid_avbd_joint_alpha=0.8,
        rigid_avbd_contact_alpha=0.1,
        rigid_avbd_linear_beta=2.0e4,
        rigid_avbd_angular_beta=3.0e3,
        rigid_avbd_gamma=0.99,
        rigid_contact_hard=True,
        rigid_contact_history=False,
        rigid_body_contact_buffer_size=256,
        rigid_joint_linear_ke=2.0e5,
        rigid_joint_angular_ke=3.0e5,
        rigid_joint_linear_kd=10.0,
        rigid_joint_angular_kd=20.0,
    )

    kwargs = NewtonManager._filter_solver_kwargs(SolverVBD, solver_cfg)

    assert kwargs["friction_epsilon"] == 0.02
    assert kwargs["rigid_avbd_alpha"] == 0.9
    assert kwargs["rigid_avbd_joint_alpha"] == 0.8
    assert kwargs["rigid_avbd_contact_alpha"] == 0.1
    assert kwargs["rigid_avbd_linear_beta"] == 2.0e4
    assert kwargs["rigid_avbd_angular_beta"] == 3.0e3
    assert kwargs["rigid_avbd_gamma"] == 0.99
    assert kwargs["rigid_contact_hard"] is True
    assert kwargs["rigid_contact_history"] is False
    assert kwargs["rigid_body_contact_buffer_size"] == 256
    assert kwargs["rigid_joint_linear_ke"] == 2.0e5
    assert kwargs["rigid_joint_angular_ke"] == 3.0e5
    assert kwargs["rigid_joint_linear_kd"] == 10.0
    assert kwargs["rigid_joint_angular_kd"] == 20.0


@pytest.mark.parametrize(
    ("soft_contact_cfg", "expected"),
    [
        pytest.param(None, (7.0, 8.0, 9.0), id="preserve"),
        pytest.param(
            NewtonSoftContactCfg(soft_contact_ke=11.0, soft_contact_kd=12.0, soft_contact_mu=13.0),
            (11.0, 12.0, 13.0),
            id="override",
        ),
    ],
)
def test_soft_contact_cfg_updates_finalized_model(monkeypatch, soft_contact_cfg, expected):
    """Soft-contact configuration updates the finalized model when provided."""
    state_values = []

    class Model:
        soft_contact_ke = 7.0
        soft_contact_kd = 8.0
        soft_contact_mu = 9.0
        world_count = 0
        articulation_count = 0

        def set_gravity(self, gravity):
            pass

        def state(self):
            state_values.append((self.soft_contact_ke, self.soft_contact_kd, self.soft_contact_mu))
            return object()

        def control(self):
            return object()

    class Builder:
        body_label = ()
        up_axis = None

        def finalize(self, *, device):
            return model

    model = Model()
    monkeypatch.setattr(PhysicsManager, "_cfg", NewtonCfg(soft_contact_cfg=soft_contact_cfg), raising=False)
    monkeypatch.setattr(PhysicsManager, "_device", "cpu", raising=False)
    monkeypatch.setattr(NewtonManager, "_builder", Builder(), raising=False)
    monkeypatch.setattr(NewtonManager, "_up_axis", "Z", raising=False)
    monkeypatch.setattr(NewtonManager, "_gravity_vector", (0.0, 0.0, -9.81), raising=False)
    monkeypatch.setattr(NewtonManager, "_num_envs", 0, raising=False)
    monkeypatch.setattr(NewtonManager, "_clone_physics_only", True, raising=False)
    monkeypatch.setattr(NewtonManager, "_pending_extended_state_attributes", set(), raising=False)
    monkeypatch.setattr(NewtonManager, "_pending_extended_contact_attributes", set(), raising=False)
    for attr in (
        "_model",
        "_state_0",
        "_state_1",
        "_control",
        "_adapter",
        "_use_newton_actuators_active",
        "_world_reset_mask",
        "_fk_reset_mask",
    ):
        monkeypatch.setattr(NewtonManager, attr, getattr(NewtonManager, attr, None), raising=False)
    monkeypatch.setattr(NewtonManager, "_cl_pending_sites", {}, raising=False)
    monkeypatch.setattr(NewtonManager, "_drain_stale_cuda_error", classmethod(lambda cls: None))
    monkeypatch.setattr(NewtonManager, "dispatch_event", classmethod(lambda cls, event: None))

    NewtonManager.start_simulation()

    assert (model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu) == expected

    assert state_values == [expected, expected]


@pytest.mark.parametrize("env_paths", [(), ("/World/Env_0", "/World/Env_1")], ids=["flat", "replicated"])
def test_vbd_excludes_registered_deformable_meshes(monkeypatch, env_paths):
    """VBD excludes registered simulation and visual meshes from USD import."""
    physics = importlib.import_module("isaaclab_newton.physics")
    pxr = importlib.import_module("pxr")
    newton_module = importlib.import_module("isaaclab_newton.physics.newton_manager")
    builders = []
    hook_calls = []
    replicate_calls = []

    class Builder:
        def __init__(self):
            self.imports = []
            self.color_calls = 0

        def add_usd(self, stage, *, root_path=None, ignore_paths=(), schema_resolvers=()):
            self.imports.append((root_path, list(ignore_paths)))
            return {"path_shape_map": {}}

        def color(self):
            self.color_calls += 1

    children = [
        SimpleNamespace(
            GetName=lambda path=path: path.rsplit("/", 1)[-1],
            GetPath=lambda path=path: SimpleNamespace(pathString=path),
        )
        for path in env_paths
    ]
    world_prim = SimpleNamespace(IsValid=lambda: True, GetChildren=lambda: children)
    stage = SimpleNamespace(GetPrimAtPath=lambda path: world_prim if path == "/World" else path)
    rotation = SimpleNamespace(GetImaginary=lambda: (0.0, 0.0, 0.0), GetReal=lambda: 1.0)
    matrix = SimpleNamespace(ExtractTranslation=lambda: (0.0, 0.0, 0.0), ExtractRotationQuat=lambda: rotation)
    usd_geom = SimpleNamespace(
        GetStageUpAxis=lambda stage: "Z",
        XformCache=lambda: SimpleNamespace(GetLocalToWorldTransform=lambda prim: matrix),
    )

    def create_builder(cls, *, up_axis):
        builder = Builder()
        builders.append(builder)
        return builder

    def replicate(*args, **kwargs):
        replicate_calls.append(kwargs)
        return {}, [object() for _ in env_paths]

    monkeypatch.setattr(newton_module, "get_current_stage", lambda: stage)
    monkeypatch.setattr(pxr, "UsdGeom", usd_geom)
    monkeypatch.setattr(newton_module, "_restore_visible_colliders_without_visual_shapes", lambda *args: None)
    monkeypatch.setattr(newton_module, "replace_newton_builder_shape_colors", lambda *args: None)
    monkeypatch.setattr(newton_module, "replicate_builder_mapping", replicate)
    monkeypatch.setattr(physics.NewtonVBDManager, "create_builder", classmethod(create_builder))
    monkeypatch.setattr(
        physics.NewtonVBDManager,
        "_inject_terrain_heightfields",
        classmethod(lambda cls, stage, builder: ["/World/terrain"]),
    )
    monkeypatch.setattr(
        physics.NewtonVBDManager,
        "_cl_inject_sites",
        classmethod(lambda cls, builder, source_builders: ({}, {}, {})),
    )
    monkeypatch.setattr(
        physics.NewtonVBDManager, "set_builder", classmethod(lambda cls, builder: setattr(cls, "_builder", builder))
    )

    def hook(builder, world_idx, position, rotation):
        hook_calls.append(world_idx)

    monkeypatch.setattr(physics.NewtonVBDManager, "_per_world_builder_hooks", [hook])
    monkeypatch.setattr(
        physics.NewtonVBDManager,
        "_deformable_registry",
        [SimpleNamespace(sim_mesh_prim_path="/World/soft/sim", vis_mesh_prim_path="/World/soft/visual")],
    )
    monkeypatch.setattr(physics.NewtonVBDManager, "_builder", None)
    monkeypatch.setattr(NewtonManager, "_cl_site_index_map", {})
    monkeypatch.setattr(NewtonManager, "_world_xforms", [])
    monkeypatch.setattr(NewtonManager, "_num_envs", 0)

    physics.NewtonVBDManager.instantiate_builder_from_stage()

    deformable_paths = ["/World/soft/sim", "/World/soft/visual"]
    if env_paths:
        assert builders[0].imports == [(None, [*env_paths, "/World/terrain", *deformable_paths])]
        assert builders[1].imports == [("/World/Env_0", deformable_paths)]
        assert replicate_calls[0]["per_world_builder_hooks"] == [hook]
    else:
        assert builders[0].imports == [(None, ["/World/terrain", *deformable_paths])]
        assert hook_calls == [0]
    assert builders[0].color_calls == 1


def test_vbd_colors_prebuilt_builder_before_start(monkeypatch):
    """VBD colors a prebuilt builder before starting simulation."""
    physics = importlib.import_module("isaaclab_newton.physics")
    deformable_module = importlib.import_module("isaaclab_contrib.deformable.deformable_object")
    events = []

    class Builder:
        def color(self):
            events.append("color")

    monkeypatch.setattr(physics.NewtonVBDManager, "_builder", Builder())
    monkeypatch.setattr(NewtonManager, "start_simulation", classmethod(lambda cls: events.append("start")))
    monkeypatch.setattr(deformable_module, "setup_registered_deformable_fabric_sync", lambda manager_cls: None)

    physics.NewtonVBDManager.start_simulation()

    assert events == ["color", "start"]


@pytest.mark.parametrize("external_rigid_solver", [False, True])
def test_vbd_solver_force_input_capability(monkeypatch, external_rigid_solver):
    """VBD accepts rigid forces only when it integrates rigid bodies."""
    physics = importlib.import_module("isaaclab_newton.physics")
    solver = object()
    monkeypatch.setattr(physics.NewtonVBDManager, "_create_solver", lambda model, cfg: solver)
    monkeypatch.setattr(physics.NewtonVBDManager, "_initialize_contacts", lambda: None)
    monkeypatch.setattr(NewtonManager, "_solver", None)
    monkeypatch.setattr(NewtonManager, "_use_single_state", True)
    monkeypatch.setattr(NewtonManager, "_needs_collision_pipeline", False)
    monkeypatch.setattr(NewtonManager, "_supports_rigid_body_force_input", False)

    solver_cfg = physics.VBDSolverCfg(integrate_with_external_rigid_solver=external_rigid_solver)
    physics.NewtonVBDManager._build_solver(object(), solver_cfg)

    assert NewtonManager._solver is solver
    assert NewtonManager._supports_rigid_body_force_input is not external_rigid_solver


def test_vbd_initializes_contacts_before_constructing_solver(monkeypatch):
    """The VBD solver should observe pipeline-published capacity on the active model."""
    physics = importlib.import_module("isaaclab_newton.physics")
    events = []
    solver = object()
    model = SimpleNamespace(rigid_contact_max=0)

    def initialize_contacts():
        events.append("contacts")
        assert NewtonManager._model is model
        model.rigid_contact_max = 4096
        NewtonManager._contacts = SimpleNamespace(rigid_contact_max=4096)

    def create_solver(input_model, cfg):
        events.append("solver")
        assert input_model is model
        assert input_model.rigid_contact_max == 4096
        assert NewtonManager._contacts.rigid_contact_max == 4096
        return solver

    monkeypatch.setattr(physics.NewtonVBDManager, "_initialize_contacts", initialize_contacts)
    monkeypatch.setattr(physics.NewtonVBDManager, "_create_solver", create_solver)
    monkeypatch.setattr(NewtonManager, "_model", model)
    monkeypatch.setattr(NewtonManager, "_solver", None)
    monkeypatch.setattr(NewtonManager, "_contacts", None)
    monkeypatch.setattr(NewtonManager, "_needs_collision_pipeline", False)
    monkeypatch.setattr(
        NewtonManager,
        "_collision_cfg",
        physics.NewtonCollisionPipelineCfg(contact_matching="latest"),
    )

    physics.NewtonVBDManager._build_solver(model, physics.VBDSolverCfg(rigid_contact_history=True))

    assert events == ["contacts", "solver"]
    assert NewtonManager._solver is solver


def test_vbd_rejects_contact_history_without_matching_before_allocation(monkeypatch):
    """Contact history without matching should fail before pipeline allocation."""
    physics = importlib.import_module("isaaclab_newton.physics")
    events = []

    monkeypatch.setattr(physics.NewtonVBDManager, "_initialize_contacts", lambda: events.append("contacts"))
    monkeypatch.setattr(NewtonManager, "_collision_cfg", physics.NewtonCollisionPipelineCfg())

    with pytest.raises(ValueError, match="contact_matching"):
        physics.NewtonVBDManager._build_solver(
            object(),
            physics.VBDSolverCfg(rigid_contact_history=True),
        )

    assert events == []


@pytest.mark.parametrize("joint_count", [0, 3])
def test_vbd_recovers_joint_state_after_solver_step(monkeypatch, joint_count):
    """VBD recovers generalized coordinates after solving maximal body state."""
    physics = importlib.import_module("isaaclab_newton.physics")
    vbd_module = importlib.import_module("isaaclab_newton.physics.vbd_manager")
    events = []
    state_0 = object()
    state_1 = SimpleNamespace(joint_q=object(), joint_qd=object())
    control = object()
    contacts = object()
    model = SimpleNamespace(joint_count=joint_count)

    class Solver:
        def step(self, input_state, output_state, input_control, input_contacts, dt):
            events.append(("step", input_state, output_state, input_control, input_contacts, dt))

    def recover_joint_state(input_model, input_state, joint_q, joint_qd):
        events.append(("eval_ik", input_model, input_state, joint_q, joint_qd))

    monkeypatch.setattr(NewtonManager, "_solver", Solver())
    monkeypatch.setattr(physics.NewtonVBDManager, "_model", model)
    monkeypatch.setattr(vbd_module, "eval_ik", recover_joint_state)

    physics.NewtonVBDManager._step_solver(state_0, state_1, control, contacts, 0.001)

    assert events[0] == ("step", state_0, state_1, control, contacts, 0.001)
    if joint_count:
        assert events[1] == ("eval_ik", model, state_1, state_1.joint_q, state_1.joint_qd)
    else:
        assert len(events) == 1


def test_vbd_rebuilds_particle_bvh_before_physics_step(monkeypatch):
    """VBD rebuilds its particle BVH before the base physics step."""
    physics = importlib.import_module("isaaclab_newton.physics")
    events = []
    state = object()

    class Solver:
        def rebuild_bvh(self, solver_state):
            events.append(("rebuild", solver_state))

    def simulate_physics_only(cls):
        events.append(("step", cls))

    monkeypatch.setattr(NewtonManager, "_simulate_physics_only", classmethod(simulate_physics_only))
    monkeypatch.setattr(physics.NewtonVBDManager, "_model", SimpleNamespace(particle_count=1))
    monkeypatch.setattr(physics.NewtonVBDManager, "_solver", Solver())
    monkeypatch.setattr(physics.NewtonVBDManager, "_state_0", state)

    physics.NewtonVBDManager._simulate_physics_only()

    assert events == [("rebuild", state), ("step", physics.NewtonVBDManager)]


def test_vbd_forwards_compliant_alm_to_compatible_newton(monkeypatch):
    """The compliant-ALM selection should reach a supporting Newton solver."""
    physics = importlib.import_module("isaaclab_newton.physics")
    vbd_module = importlib.import_module("isaaclab_newton.physics.vbd_manager")

    class CompatibleSolver:
        def __init__(self, model, *, rigid_compliant_alm=None):
            self.model = model
            self.rigid_compliant_alm = rigid_compliant_alm

    monkeypatch.setattr(vbd_module, "SolverVBD", CompatibleSolver)

    model = object()
    solver = physics.NewtonVBDManager._create_solver(
        model,
        physics.VBDSolverCfg(rigid_compliant_alm=True),
    )

    assert solver.model is model
    assert solver.rigid_compliant_alm is True


def test_vbd_preserves_legacy_newton_default(monkeypatch):
    """The default should omit compliant ALM for a legacy Newton constructor."""
    physics = importlib.import_module("isaaclab_newton.physics")
    vbd_module = importlib.import_module("isaaclab_newton.physics.vbd_manager")

    class LegacySolver:
        def __init__(self, model):
            self.model = model

    monkeypatch.setattr(vbd_module, "SolverVBD", LegacySolver)

    model = object()
    solver = physics.NewtonVBDManager._create_solver(
        model,
        physics.VBDSolverCfg(rigid_compliant_alm=None),
    )

    assert solver.model is model


def test_vbd_rejects_compliant_alm_before_contact_allocation(monkeypatch):
    """Unsupported compliant ALM should fail before allocating contact buffers."""
    physics = importlib.import_module("isaaclab_newton.physics")
    vbd_module = importlib.import_module("isaaclab_newton.physics.vbd_manager")
    events = []

    class LegacySolver:
        def __init__(self, model):
            self.model = model

    monkeypatch.setattr(vbd_module, "SolverVBD", LegacySolver)
    monkeypatch.setattr(physics.NewtonVBDManager, "_initialize_contacts", lambda: events.append("contacts"))

    with pytest.raises(RuntimeError, match="rigid_compliant_alm"):
        physics.NewtonVBDManager._build_solver(
            object(),
            physics.VBDSolverCfg(rigid_compliant_alm=True),
        )

    assert events == []
