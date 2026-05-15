#!/usr/bin/env python3
"""Create an SDF-preserving culled gear asset variant from USD gear assets.

The script copies an input gear asset tree, cleans each UsdGeom.Mesh, applies
Open3D quadric decimation, preserves authored USD physics collision APIs, and
writes a JSON manifest with face-count and collision-approximation checks. It is
intended for producing lower-triangle collision/source meshes that still cook as
SDF at runtime, rather than switching the task to convex hull collision.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import open3d as o3d
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

DEFAULT_ASSET_ROOT = (
    Path(__file__).resolve().parents[2] / "source/isaaclab_tasks/isaaclab_tasks/direct/forge/assets"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_triangle_mesh(face_counts) -> bool:
    return bool(face_counts) and all(int(c) == 3 for c in face_counts)


def open3d_mesh_from_usd(mesh: UsdGeom.Mesh) -> o3d.geometry.TriangleMesh:
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    face_counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
    face_indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32, copy=True)
    if not is_triangle_mesh(face_counts):
        raise ValueError(f"Only triangle meshes are supported for {mesh.GetPath()}; face counts include {sorted(set(face_counts))[:8]}")
    triangles = face_indices.reshape((-1, 3)).copy()
    out = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(points),
        triangles=o3d.utility.Vector3iVector(triangles),
    )
    out.remove_duplicated_vertices()
    out.remove_duplicated_triangles()
    out.remove_degenerate_triangles()
    out.remove_unreferenced_vertices()
    return out


def write_open3d_to_usd(mesh: UsdGeom.Mesh, reduced: o3d.geometry.TriangleMesh) -> None:
    reduced.remove_duplicated_vertices()
    reduced.remove_duplicated_triangles()
    reduced.remove_degenerate_triangles()
    reduced.remove_unreferenced_vertices()
    verts = np.asarray(reduced.vertices, dtype=np.float32)
    tris = np.asarray(reduced.triangles, dtype=np.int64)
    mesh.GetPointsAttr().Set([tuple(map(float, p)) for p in verts])
    mesh.GetFaceVertexCountsAttr().Set([3] * len(tris))
    mesh.GetFaceVertexIndicesAttr().Set([int(i) for tri in tris for i in tri])
    # Avoid stale authored normals/primvars after topology changes. Kit/PhysX can recompute.
    for attr_name in ["normals", "primvars:normals"]:
        attr = mesh.GetPrim().GetAttribute(attr_name)
        if attr:
            attr.Clear()


def collision_approximations(stage: Usd.Stage) -> list[dict]:
    rows = []
    for prim in stage.TraverseAll():
        mesh_api = UsdPhysics.MeshCollisionAPI(prim)
        if mesh_api:
            rows.append(
                {
                    "path": str(prim.GetPath()),
                    "type": prim.GetTypeName(),
                    "approximation": mesh_api.GetApproximationAttr().Get(),
                    "applied_schemas": list(prim.GetAppliedSchemas()),
                }
            )
    return rows


def decimate_file(path: Path, keep_factor: float, min_faces: int) -> dict:
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {path}")
    file_record = {
        "file": str(path),
        "collision_approximations_before": collision_approximations(stage),
        "meshes": [],
    }
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        orig_points = mesh.GetPointsAttr().Get() or []
        orig_face_counts = mesh.GetFaceVertexCountsAttr().Get() or []
        orig_faces = len(orig_face_counts)
        if orig_faces == 0:
            continue
        target_faces = max(min_faces, int(round(orig_faces * keep_factor)))
        target_faces = min(target_faces, orig_faces)
        o3mesh = open3d_mesh_from_usd(mesh)
        clean_faces = len(o3mesh.triangles)
        clean_verts = len(o3mesh.vertices)
        before_watertight = bool(o3mesh.is_watertight())
        if target_faces < clean_faces:
            reduced = o3mesh.simplify_quadric_decimation(target_faces)
            reduced.remove_duplicated_vertices()
            reduced.remove_duplicated_triangles()
            reduced.remove_degenerate_triangles()
            reduced.remove_unreferenced_vertices()
        else:
            reduced = o3mesh
        after_watertight = bool(reduced.is_watertight())
        write_open3d_to_usd(mesh, reduced)
        file_record["meshes"].append(
            {
                "path": str(prim.GetPath()),
                "orig_verts": len(orig_points),
                "orig_faces": orig_faces,
                "clean_verts": clean_verts,
                "clean_faces": clean_faces,
                "target_faces": target_faces,
                "new_verts": len(reduced.vertices),
                "new_faces": len(reduced.triangles),
                "watertight_before": before_watertight,
                "watertight_after": after_watertight,
            }
        )
    file_record["collision_approximations_after"] = collision_approximations(stage)
    stage.GetRootLayer().Save()
    return file_record


def scan_counts(root: Path) -> dict:
    totals = {"meshes": 0, "verts": 0, "faces": 0}
    files = []
    for path in sorted(root.rglob("*.usd")):
        stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
        if stage is None:
            continue
        meshes = []
        collisions = []
        for prim in stage.TraverseAll():
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                pts = mesh.GetPointsAttr().Get() or []
                faces = mesh.GetFaceVertexCountsAttr().Get() or []
                meshes.append({"path": str(prim.GetPath()), "verts": len(pts), "faces": len(faces)})
                totals["meshes"] += 1
                totals["verts"] += len(pts)
                totals["faces"] += len(faces)
            mesh_api = UsdPhysics.MeshCollisionAPI(prim)
            if mesh_api:
                collisions.append({"path": str(prim.GetPath()), "approximation": mesh_api.GetApproximationAttr().Get()})
        if meshes or collisions:
            files.append({"file": str(path.relative_to(root)), "meshes": meshes, "collisions": collisions})
    return {"totals": totals, "files": files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Input gear asset tree to copy and cull, e.g. path/to/props_factory_gear_assets or a variant root.",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=DEFAULT_ASSET_ROOT,
        help="Root directory used when --variant is relative (default: IsaacLab forge/assets).",
    )
    parser.add_argument(
        "--variant",
        required=True,
        help="Destination directory name under --asset-root, or an absolute output path.",
    )
    parser.add_argument("--keep-factor", type=float, default=0.5, help="Per-mesh face keep factor relative to source")
    parser.add_argument("--min-faces", type=int, default=96)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    variant_path = Path(args.variant)
    dest = (variant_path if variant_path.is_absolute() else args.asset_root / variant_path).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if dest.exists():
        if not args.overwrite:
            raise FileExistsError(f"Destination exists: {dest}; use --overwrite to replace it")
        shutil.rmtree(dest)
    shutil.copytree(source, dest, symlinks=True)

    manifest = {
        "created_at": utc_now(),
        "source": str(source),
        "destination": str(dest),
        "keep_factor_from_source": args.keep_factor,
        "min_faces": args.min_faces,
        "method": "Copied input USD asset tree and applied Open3D TriangleMesh.simplify_quadric_decimation per UsdGeom.Mesh; duplicate/degenerate/unreferenced geometry cleanup; cleared normals. Existing physics MeshCollisionAPI approximation is inspected and preserved; use SDF-authored source assets to produce SDF-preserving outputs.",
        "before": scan_counts(source),
        "changed_files": [],
    }
    for path in sorted(dest.rglob("*.usd")):
        record = decimate_file(path, args.keep_factor, args.min_faces)
        if record["meshes"] or record["collision_approximations_before"]:
            manifest["changed_files"].append(record)
    manifest["after"] = scan_counts(dest)

    non_sdf = []
    for file_rec in manifest["after"]["files"]:
        for coll in file_rec["collisions"]:
            if coll.get("approximation") != "sdf":
                non_sdf.append({"file": file_rec["file"], **coll})
    manifest["non_sdf_collision_approximations"] = non_sdf
    manifest_path = dest / "manifest_reduction.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({
        "destination": str(dest),
        "manifest": str(manifest_path),
        "before_totals": manifest["before"]["totals"],
        "after_totals": manifest["after"]["totals"],
        "non_sdf_collision_approximations": non_sdf,
    }, indent=2))


if __name__ == "__main__":
    main()
