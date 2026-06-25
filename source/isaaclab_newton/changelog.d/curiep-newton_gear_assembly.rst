Fixed
^^^^^

* Fixed the Newton cloner to recover mesh colliders that ``ModelBuilder.add_usd`` skips when
  ``UsdPhysics.CollisionAPI`` is authored on a parent ``Xform`` over the collision mesh (e.g.
  the factory gears), so those bodies collide instead of falling through everything.
* Added authoring of real approximation-tagged (``sdf``) colliders for such bodies when
  Newton's own collision pipeline is active, preserving concave geometry for hydroelastic
  contacts instead of collapsing it to a convex hull.
* Kept gripper finger colliders out of the convex-hull simplification so their concave contact
  surface conforms to a grasped object instead of a convex shell that interpenetrates it (affects
  Newton's own collision pipeline; MuJoCo contacts convex-hull at solve time regardless).
* Welded duplicate (coincident) vertices on collision meshes before the Newton collision pipeline
  builds their signed-distance fields. Meshes exported from CAD tools are often not watertight,
  which PhysX tolerates but leaves the SDF ill-defined, ejecting or interpenetrating grasped and
  seated objects. Welding closes the seams without changing the surface geometry.
* Enabled Newton hydroelastic SDF collision setup for recovered mesh colliders by building
  mesh SDFs and setting hydroelastic shape flags when ``sdf_hydroelastic_config`` is
  active.
* Preserved directly authored ``sdf`` mesh colliders during convex-hull simplification so
  package-authored Newton gear and base colliders keep their concave collision geometry.
