Fixed
^^^^^

* Fixed missing DisplayPort socket shell visuals in the Newton viewer by separating
  visual meshes from SDF collision meshes.
* Fixed unstable DisplayPort plug grasps in Newton by calibrating the fingertip
  pose, clamp width, contact friction, and Grav gripper actuator gains.
* Improved full-depth DisplayPort insertion in Newton with higher-resolution
  mesh SDFs while preserving the exact seated target used by PhysX.
