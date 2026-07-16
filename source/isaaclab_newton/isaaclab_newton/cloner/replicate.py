# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import torch
from newton import ModelBuilder
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx

from pxr import Usd

from isaaclab.cloner.replicate_session import REPLICATION_QUEUE
from isaaclab.physics import PhysicsManager
from isaaclab.sim.utils.newton_model_utils import replace_newton_builder_shape_colors

from isaaclab_newton.cloner.newton_clone_utils import (
    build_source_builders,
    rename_builder_labels,
    replicate_builder_mapping,
)
from isaaclab_newton.physics import NewtonManager

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    _MappingBatch: TypeAlias = tuple[
        tuple[str, ...], tuple[str, ...], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None
    ]
else:
    _MappingBatch = tuple


def _enable_intended_mesh_colliders(builder: ModelBuilder, stage: Usd.Stage) -> list[int]:
    """Enable collision on mesh shapes whose USD body declares collision intent but for which
    Newton's importer produced no collider.

    Newton's :meth:`ModelBuilder.add_usd` only creates a collider when ``UsdPhysics.CollisionAPI``
    is applied directly to a geometry prim. Some assets (e.g. the factory gears) apply the API to a
    parent ``Xform`` over the collision mesh, so the importer skips the collider and loads only the
    (identical-geometry) visual mesh as a non-colliding shape -- the body then has no collision and
    falls through everything. For each body that has shapes but none colliding, if its USD subtree
    declares collision intent, enable ``COLLIDE_SHAPES`` on its mesh shapes, reusing the geometry and
    transform already imported for the visual mesh.

    Must run on the builder before :meth:`ModelBuilder.finalize` (here, on each prototype before
    replication).

    Args:
        builder: The prototype model builder to repair in place.
        stage: The USD stage the builder was populated from.

    Returns:
        Indices of the shapes switched to colliding. These are full (non-remeshed) meshes, so the
        caller should keep them out of any convex-hull approximation pass to preserve the concave
        geometry the assets are authored for (e.g. ``sdf``).
    """
    from newton import GeoType, ShapeFlags

    from pxr import UsdPhysics

    collide_bit = int(ShapeFlags.COLLIDE_SHAPES)
    mesh_type = int(GeoType.MESH)

    body_to_shapes: dict[int, list[int]] = {}
    for i in range(builder.shape_count):
        body_to_shapes.setdefault(builder.shape_body[i], []).append(i)

    recovered: list[int] = []
    for body_idx, shape_ids in body_to_shapes.items():
        if body_idx < 0:
            continue
        if any(int(builder.shape_flags[i]) & collide_bit for i in shape_ids):
            continue  # body already has a collider; nothing was skipped
        prim = stage.GetPrimAtPath(builder.body_label[body_idx])
        if not prim or not prim.IsValid():
            continue
        if not any(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())):
            continue
        for i in shape_ids:
            if int(builder.shape_type[i]) == mesh_type:
                builder.shape_flags[i] = int(builder.shape_flags[i]) | collide_bit
                recovered.append(i)
    return recovered


def _author_skipped_mesh_colliders(stage: Usd.Stage, sources: Sequence[str]) -> set[str]:
    """Author colliders the importer would otherwise skip, as real (approximation-tagged) colliders.

    Newton's :meth:`ModelBuilder.add_usd` only builds a collider when ``UsdPhysics.CollisionAPI`` is
    applied directly to a geometry prim. Assets such as the factory gears apply the API to a parent
    ``Xform`` over an *instanced* collision mesh, so the importer builds no collider at all. De-instance
    those subtrees and apply ``CollisionAPI`` -- plus the ancestor's ``MeshCollisionAPI`` approximation
    (e.g. ``sdf``) -- to the collision mesh itself, so the importer builds a real collider that honors
    the authored approximation. This is what lets the hydroelastic SDF path grip concave geometry (such
    as a gear hub) instead of a filled-in convex hull.

    Must run on the stage before any :meth:`ModelBuilder.add_usd`.

    Args:
        stage: USD stage to edit in place.
        sources: Source prim paths to scan.

    Returns:
        Rigid-body prim paths for which a collider was authored. The caller keeps these bodies'
        collision meshes out of convex-hull approximation to preserve their concave geometry.
    """
    from pxr import UsdPhysics

    authored: set[str] = set()
    for src in sources:
        root = stage.GetPrimAtPath(src)
        if not root or not root.IsValid():
            continue
        root_path = root.GetPath()
        # De-instance so proxy collision meshes become real, authorable prims.
        for prim in Usd.PrimRange(root):
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
        for prim in Usd.PrimRange(root):
            if prim.GetTypeName() != "Mesh" or prim.HasAPI(UsdPhysics.CollisionAPI):
                continue  # not a mesh, or the importer already builds this collider
            # Walk up to the nearest ancestor that declares collision intent on an Xform.
            approx = None
            declares = False
            ancestor = prim.GetParent()
            while ancestor and ancestor.IsValid() and ancestor.GetPath().HasPrefix(root_path):
                if ancestor.HasAPI(UsdPhysics.CollisionAPI):
                    declares = True
                    if ancestor.HasAPI(UsdPhysics.MeshCollisionAPI):
                        approx = UsdPhysics.MeshCollisionAPI(ancestor).GetApproximationAttr().Get()
                    break
                ancestor = ancestor.GetParent()
            if not declares:
                continue
            UsdPhysics.CollisionAPI.Apply(prim)
            mca = UsdPhysics.MeshCollisionAPI.Apply(prim)
            if approx:
                mca.CreateApproximationAttr().Set(approx)
            # Record the owning rigid body (nearest ancestor with RigidBodyAPI, else the source root).
            body = prim
            while body and body.IsValid() and body.GetPath().HasPrefix(root_path):
                if body.HasAPI(UsdPhysics.RigidBodyAPI):
                    break
                body = body.GetParent()
            authored.add(str(body.GetPath()) if body and body.IsValid() else str(root_path))
    return authored


def _weld_builder_collision_meshes(source_builders: dict[str, ModelBuilder]) -> int:
    """Merge coincident (duplicate) vertices on mesh colliders so they are watertight.

    Meshes exported from CAD tools (e.g. Onshape) often emit per-face duplicate vertices, leaving the
    surface topologically open (non-watertight). PhysX tolerates this, but Newton's own collision
    pipeline builds a signed-distance field from the mesh, which is only well defined for a watertight
    surface -- otherwise contacts are spurious and grasped/seated objects are ejected or
    interpenetrated. Welding coincident vertices closes these seams without changing the surface
    geometry (it only reindexes faces onto a deduplicated vertex set).

    Runs on the per-source :class:`ModelBuilder` mesh shapes (after import and convex-hull
    simplification, before replication), so it is unaffected by USD instancing of the collision
    geometry. Only concave ``MESH`` shapes are touched -- convex-hull shapes are already watertight.

    Args:
        source_builders: Per-source builders to edit in place.

    Returns:
        Number of mesh colliders that were welded.
    """
    from newton import GeoType, Mesh, ShapeFlags

    mesh_type = int(GeoType.MESH)
    collide_bit = int(ShapeFlags.COLLIDE_SHAPES)
    welded = 0
    for builder in source_builders.values():
        for i in range(builder.shape_count):
            if int(builder.shape_type[i]) != mesh_type or not (int(builder.shape_flags[i]) & collide_bit):
                continue  # only concave colliders get an SDF; skip non-colliding (e.g. visual) meshes
            source = builder.shape_source[i]
            if source is None or getattr(source, "vertices", None) is None:
                continue
            vertices = np.asarray(source.vertices, dtype=np.float64)
            indices = np.asarray(source.indices, dtype=np.int64)
            if len(vertices) == 0 or len(indices) == 0:
                continue
            # Quantize to a small fraction of the mesh extent so only coincident duplicates merge.
            extent = float(np.max(vertices.max(axis=0) - vertices.min(axis=0)))
            tol = max(extent, 1.0) * 1e-7
            keys = np.round(vertices / tol).astype(np.int64)
            _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
            if len(first) == len(vertices):
                continue  # no duplicate vertices to merge
            new_vertices = vertices[first].astype(np.float32)
            new_indices = inverse[indices].astype(np.int32)
            builder.shape_source[i] = Mesh(new_vertices, new_indices)
            welded += 1
    return welded


def _configure_hydroelastic_sdf_shapes(source_builders: dict[str, ModelBuilder], *, max_resolution: int = 64) -> int:
    """Build SDFs and enable hydroelastic contacts on eligible colliding shapes.

    Newton hydroelastic SDF contact is activated per shape: both shapes in a contact pair must carry
    :attr:`ShapeFlags.HYDROELASTIC`, and mesh shapes must have an SDF attached before model
    finalization. USD collision schemas declare the author's intent, but Newton's builder still needs
    explicit mesh SDF data and hydroelastic flags.

    Args:
        source_builders: Per-source builders to edit in place.
        max_resolution: Maximum generated SDF grid dimension. Must be positive
            and divisible by 8.

    Returns:
        Number of shapes marked as hydroelastic.
    """
    from newton import GeoType, ShapeFlags

    if max_resolution <= 0 or max_resolution % 8 != 0:
        raise ValueError(f"max_resolution must be positive and divisible by 8, got {max_resolution}.")

    collide_bit = int(ShapeFlags.COLLIDE_SHAPES)
    hydroelastic_bit = int(ShapeFlags.HYDROELASTIC)
    mesh_types = {int(GeoType.MESH), int(GeoType.CONVEX_MESH)}
    primitive_types = {
        int(GeoType.BOX),
        int(GeoType.SPHERE),
        int(GeoType.CAPSULE),
        int(GeoType.CYLINDER),
        int(GeoType.CONE),
        int(GeoType.ELLIPSOID),
    }
    marked = 0
    for builder in source_builders.values():
        for i in range(builder.shape_count):
            if not (int(builder.shape_flags[i]) & collide_bit):
                continue
            shape_type = int(builder.shape_type[i])
            source = builder.shape_source[i]
            if shape_type in mesh_types and source is not None and getattr(source, "vertices", None) is not None:
                shape_scale = np.asarray(builder.shape_scale[i], dtype=np.float32)
                mesh = source
                if not np.allclose(shape_scale, 1.0):
                    mesh = mesh.copy(vertices=np.asarray(mesh.vertices, dtype=np.float32) * shape_scale)
                    builder.shape_source[i] = mesh
                    builder.shape_scale[i] = (1.0, 1.0, 1.0)
                if getattr(mesh, "sdf", None) is None:
                    gap = float(builder.shape_gap[i] if builder.shape_gap[i] is not None else 0.005)
                    sdf_radius = max(gap, 0.005)
                    mesh.build_sdf(
                        max_resolution=max_resolution,
                        narrow_band_range=(-sdf_radius, sdf_radius),
                        margin=sdf_radius,
                    )
                builder.shape_type[i] = GeoType.MESH
            elif shape_type not in primitive_types:
                continue
            builder.shape_flags[i] = int(builder.shape_flags[i]) | hydroelastic_bit
            marked += 1
    return marked


def _find_direct_sdf_mesh_colliders(builder: ModelBuilder, stage: Usd.Stage) -> set[int]:
    """Find imported mesh colliders whose USD mesh explicitly requests ``sdf`` collision.

    These meshes have already been imported as real colliders, so they do not go through
    :func:`_author_skipped_mesh_colliders`. They still need the same protection from the convex-hull
    simplifier; otherwise an authored SDF gear/base collider is replaced by a filled-in hull before
    hydroelastic SDF contacts are built.
    """
    from newton import GeoType, ShapeFlags

    from pxr import UsdPhysics

    collide_bit = int(ShapeFlags.COLLIDE_SHAPES)
    mesh_type = int(GeoType.MESH)
    sdf_shape_ids: set[int] = set()
    for i in range(builder.shape_count):
        if int(builder.shape_type[i]) != mesh_type or not (int(builder.shape_flags[i]) & collide_bit):
            continue
        shape_label = str(builder.shape_label[i])
        prim = stage.GetPrimAtPath(shape_label)
        if not prim or not prim.IsValid() or not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            continue
        approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        if str(approximation) == "sdf":
            sdf_shape_ids.add(i)
    return sdf_shape_ids


def _recover_and_simplify_source_builders(
    source_builders: dict[str, ModelBuilder],
    stage: Usd.Stage,
    authored_bodies: set[str],
    simplify_meshes: bool,
    preserve_sdf_meshes: bool = False,
) -> None:
    """Recover importer-skipped colliders and convex-hull the rest, in place on each source builder.

    Runs after :func:`build_source_builders` (called with ``simplify_meshes=False``). For each source
    builder it enables collision on mesh shapes the importer skipped (see
    :func:`_enable_intended_mesh_colliders`), then convex-hulls every remaining collision mesh except
    those recovered shapes and the bodies given real (e.g. ``sdf``) colliders by
    :func:`_author_skipped_mesh_colliders` -- both must keep their concave geometry. When
    ``preserve_sdf_meshes`` is true, directly authored ``sdf`` mesh colliders are kept as well.
    """
    if not simplify_meshes:
        return
    from newton import GeoType, ShapeFlags

    collide_bit = int(ShapeFlags.COLLIDE_SHAPES)
    mesh_type = int(GeoType.MESH)
    for p in source_builders.values():
        recovered = _enable_intended_mesh_colliders(p, stage)
        keep = set(recovered)
        keep.update(i for i in range(p.shape_count) if p.body_label[p.shape_body[i]] in authored_bodies)
        if preserve_sdf_meshes:
            keep.update(_find_direct_sdf_mesh_colliders(p, stage))
        # Keep gripper finger colliders concave (out of the hull) so the contact surface conforms to a
        # grasped object instead of a convex shell that interpenetrates it. The MuJoCo-contacts preset
        # convex-hulls at solve time regardless, so this only affects Newton's own (e.g. hydroelastic)
        # collision pipeline, where it matches the concave PhysX gripper collision.
        keep.update(i for i in range(p.shape_count) if "finger" in str(p.body_label[p.shape_body[i]]).lower())
        hull_ids = [
            i
            for i in range(p.shape_count)
            if i not in keep and int(p.shape_type[i]) == mesh_type and (int(p.shape_flags[i]) & collide_bit)
        ]
        if hull_ids:
            p.approximate_meshes("convex_hull", shape_indices=hull_ids, keep_visual_shapes=True)


def _build_newton_builder_from_mapping(
    stage: Usd.Stage,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
    up_axis: str = "Z",
    simplify_meshes: bool = True,
) -> tuple[ModelBuilder, object, dict, list, dict[str, ModelBuilder]]:
    """Build a Newton model builder from clone mapping inputs.

    Also returns the per-source builders (``{source_path: ModelBuilder}``) so the
    committing path can retain them for single-model consumers such as the
    batched Newton IK action.
    """
    if positions is None:
        positions = torch.zeros((mapping.size(1), 3), device=mapping.device, dtype=torch.float32)
    if quaternions is None:
        quaternions = torch.zeros((mapping.size(1), 4), device=mapping.device, dtype=torch.float32)
        quaternions[:, 3] = 1.0

    schema_resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]
    manager_cls = PhysicsManager._sim.physics_manager

    builder = manager_cls.create_builder(up_axis=up_axis)
    stage_info = builder.add_usd(
        stage,
        ignore_paths=["/World/envs", *sources],
        schema_resolvers=schema_resolvers,
    )
    replace_newton_builder_shape_colors(builder, stage)

    # Deformable prim paths are handled by per_world_builder_hooks, not add_usd.
    # Resolve the regex prim_path patterns to concrete env_0 paths so add_usd
    # can skip them via ignore_paths.
    deformable_patterns = tuple(
        re.compile(entry.prim_path.replace(".*", "[^/]*")) for entry in NewtonManager._deformable_registry
    )
    deformable_ignore_paths = []
    if deformable_patterns:
        for source in sources:
            for child in Usd.PrimRange(stage.GetPrimAtPath(source)):
                child_path = str(child.GetPath())
                if any(pattern.fullmatch(child_path) for pattern in deformable_patterns):
                    deformable_ignore_paths.append(child_path)

    # Author proper colliders for bodies whose collision mesh the importer would skip (CollisionAPI on
    # a parent Xform of an instanced mesh, e.g. the factory gears), so the importer builds real,
    # approximation-tagged (sdf) colliders the hydroelastic path can use. Only for Newton's own
    # collision pipeline (``use_mujoco_contacts=False``): under MuJoCo contacts the solver convex-hulls
    # the concave collision mesh into a degenerate hull (NaN training), so there we keep the lighter
    # ``_enable_intended_mesh_colliders`` flag-recovery on the visual mesh instead.
    # Whether Newton's own (SDF) collision pipeline will be used. ``_needs_collision_pipeline`` is only
    # set True later, when the solver is built (after this cloner runs), so at this point we detect the
    # intent from the config: a ``collision_cfg`` is authored only for the SDF preset. Fall back to the
    # flag in case it is already set.
    needs_sdf = (
        NewtonManager._needs_collision_pipeline or getattr(PhysicsManager._cfg, "collision_cfg", None) is not None
    )

    authored_bodies: set[str] = set()
    if needs_sdf:
        authored_bodies = _author_skipped_mesh_colliders(stage, sources)

    # Build source builders without simplification, then recover importer-skipped colliders before
    # convex-hulling (so recovered/authored concave meshes are preserved). See
    # :func:`_recover_and_simplify_source_builders`.
    source_builders = build_source_builders(
        stage,
        sources,
        lambda: manager_cls.create_builder(up_axis=up_axis),
        schema_resolvers,
        ignore_paths=deformable_ignore_paths or None,
        simplify_meshes=False,
    )
    _recover_and_simplify_source_builders(
        source_builders, stage, authored_bodies, simplify_meshes, preserve_sdf_meshes=needs_sdf
    )

    # Weld duplicate vertices on the remaining concave mesh colliders so the SDF pipeline sees
    # watertight surfaces (CAD exports often are not). After simplification so convex-hull shapes are
    # left alone, and on the builder (not the USD stage) so it is robust to collision-mesh instancing.
    if needs_sdf:
        welded = _weld_builder_collision_meshes(source_builders)
        if welded:
            logger.info("Welded duplicate vertices on %d mesh collider(s) for the SDF pipeline.", welded)
        collision_cfg = getattr(PhysicsManager._cfg, "collision_cfg", None)
        sdf_hydroelastic_config = getattr(collision_cfg, "sdf_hydroelastic_config", None)
        if sdf_hydroelastic_config is not None:
            marked = _configure_hydroelastic_sdf_shapes(
                source_builders,
                max_resolution=sdf_hydroelastic_config.sdf_max_resolution,
            )
            if marked:
                logger.info("Enabled hydroelastic SDF contacts on %d shape(s).", marked)

    # Inject registered sites into source builders (and global sites into main builder).
    global_sites, source_sites, root_sites = NewtonManager._cl_inject_sites(builder, source_builders)

    replicate_args = (builder, sources, mapping, positions, quaternions, source_builders)
    local_site_map, world_xforms = replicate_builder_mapping(
        *replicate_args,
        source_site_indices=source_sites,
        env_root_sites=root_sites,
        per_world_builder_hooks=NewtonManager._per_world_builder_hooks,
        post_replicate_hooks=NewtonManager._post_replicate_hooks,
    )

    site_index_map = {label: (idx, None) for label, idx in global_sites.items()}
    site_index_map.update((label, (None, per_world)) for label, per_world in local_site_map.items())
    return builder, stage_info, site_index_map, world_xforms, source_builders


class NewtonReplicateContext:
    """Queue and run Newton replication work for one stage."""

    def __init__(
        self,
        stage: Usd.Stage,
        *,
        device: str = "cpu",
        up_axis: str = "Z",
        simplify_meshes: bool | None = None,
        commit_to_manager: bool = True,
    ):
        """Initialize the context.

        Args:
            stage: USD stage containing source assets.
            device: Device used by the finalized Newton model builder.
            up_axis: Up axis for the Newton model builder.
            simplify_meshes: Whether to run convex-hull mesh approximation. If
                ``None``, read from the active :class:`NewtonCfg`.
            commit_to_manager: Whether :meth:`replicate` should publish the builder to
                :class:`NewtonManager`.
        """
        self.stage = stage
        self.device = device
        self.up_axis = up_axis
        if simplify_meshes is None:
            from isaaclab_newton.physics import NewtonCfg

            cfg = PhysicsManager._cfg
            simplify_meshes = cfg.simplify_meshes if isinstance(cfg, NewtonCfg) else True
        self.simplify_meshes = simplify_meshes
        self.commit_to_manager = commit_to_manager
        self._queue: list[_MappingBatch] = []

    def queue_mapping(
        self,
        sources: Sequence[str],
        destinations: Sequence[str],
        env_ids: torch.Tensor,
        mapping: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        quaternions: torch.Tensor | None = None,
    ) -> None:
        """Queue replication rows from the current flat clone mapping.

        Args:
            sources: Source prim paths used for cloning.
            destinations: Destination prim path templates.
            env_ids: Environment ids for destination worlds.
            mapping: Boolean source-to-environment mapping matrix.
            positions: Optional per-environment world positions [m].
            quaternions: Optional per-environment orientations in xyzw order.
        """
        self._queue.append((tuple(sources), tuple(destinations), env_ids, mapping, positions, quaternions))

    @staticmethod
    def _merge_optional_tensor(
        name: str, current: torch.Tensor | None, incoming: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Merge optional tensors, requiring equal values when both are present."""
        if current is None:
            return incoming
        if incoming is None:
            return current
        if current.device != incoming.device or current.shape != incoming.shape or not torch.equal(current, incoming):
            raise ValueError(f"Queued Newton mappings must use the same {name} tensor.")
        return current

    def _merged_mapping(self) -> _MappingBatch:
        """Merge queued mapping batches into the legacy flat mapping shape."""
        if not self._queue:
            raise RuntimeError("Cannot replicate without queued Newton mappings.")

        sources: list[str] = []
        destinations: list[str] = []
        mappings: list[torch.Tensor] = []
        env_ids = self._queue[0][2]
        positions = self._queue[0][4]
        quaternions = self._queue[0][5]

        for (
            queued_sources,
            queued_destinations,
            queued_env_ids,
            mapping,
            queued_positions,
            queued_quaternions,
        ) in self._queue:
            if (
                env_ids.device != queued_env_ids.device
                or env_ids.shape != queued_env_ids.shape
                or not torch.equal(env_ids, queued_env_ids)
            ):
                raise ValueError("Queued Newton mappings must use the same env_ids tensor.")
            sources.extend(queued_sources)
            destinations.extend(queued_destinations)
            mappings.append(mapping)
            positions = self._merge_optional_tensor("positions", positions, queued_positions)
            quaternions = self._merge_optional_tensor("quaternions", quaternions, queued_quaternions)

        return tuple(sources), tuple(destinations), env_ids, torch.cat(mappings, dim=0), positions, quaternions

    def replicate(self) -> tuple[ModelBuilder, object, dict]:
        """Build the Newton model builder from queued mappings and optionally publish it."""
        sources, destinations, env_ids, mapping, positions, quaternions = self._merged_mapping()
        builder, stage_info, site_index_map, world_xforms, source_builders = _build_newton_builder_from_mapping(
            stage=self.stage,
            sources=sources,
            destinations=destinations,
            env_ids=env_ids,
            mapping=mapping,
            positions=positions,
            quaternions=quaternions,
            up_axis=self.up_axis,
            simplify_meshes=self.simplify_meshes,
        )
        fabric_body_bindings = rename_builder_labels(builder, sources, destinations, env_ids, mapping)
        if self.commit_to_manager:
            NewtonManager._cl_site_index_map = site_index_map
            NewtonManager._cl_fabric_body_bindings = fabric_body_bindings
            NewtonManager._world_xforms = world_xforms
            NewtonManager._cl_protos = source_builders
            NewtonManager._cl_proto_models = {}
            NewtonManager.set_builder(builder)
            NewtonManager._num_envs = mapping.size(1)
        self._queue.clear()
        return builder, stage_info, site_index_map


def queue_newton_physics_replication(cfg: Any) -> None:
    """Register ``cfg`` for Newton replication when :func:`~isaaclab.cloner.replicate` next runs.

    Appends ``(cfg, NewtonReplicateContext)`` to
    :data:`~isaaclab.cloner.REPLICATION_QUEUE`. The actual row resolution and dispatch
    happen inside :func:`~isaaclab.cloner.replicate`, so this helper is safe to call from
    any asset constructor — no active session is required.
    """
    REPLICATION_QUEUE.append((cfg, NewtonReplicateContext))


def newton_physics_replicate(
    stage: Usd.Stage,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
    device: str = "cpu",
    up_axis: str = "Z",
    simplify_meshes: bool = True,
):
    """Replicate prims into a Newton ``ModelBuilder`` using a per-source mapping.

    Args:
        stage: USD stage containing source assets.
        sources: Source prim paths used for cloning.
        destinations: Destination prim path templates.
        env_ids: Environment ids for destination worlds.
        mapping: Boolean source-to-environment mapping matrix.
        positions: Optional per-environment world positions.
        quaternions: Optional per-environment orientations in xyzw order.
        device: Device used by the finalized Newton model builder.
        up_axis: Up axis for the Newton model builder.
        simplify_meshes: Whether to run convex-hull mesh approximation.

    Returns:
        Tuple of the populated Newton model builder and stage metadata.
    """
    ctx = NewtonReplicateContext(
        stage, device=device, up_axis=up_axis, simplify_meshes=simplify_meshes, commit_to_manager=True
    )
    ctx.queue_mapping(sources, destinations, env_ids, mapping, positions=positions, quaternions=quaternions)
    builder, stage_info, _site_index_map = ctx.replicate()
    return builder, stage_info
