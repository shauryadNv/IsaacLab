Added
^^^^^

* Added a Newton point-SDF operational-space-control training,
  playback, and ROS-inference profile for DisplayPort insertion.

Fixed
^^^^^

* Fixed reset-time grasp inverse kinematics to use backend-neutral articulation
  Jacobians on PhysX and Newton.

* Fixed reset-time grasp randomization to hold one target throughout each
  iterative inverse-kinematics solve.
* Fixed Newton ROS-inference metadata to expose canonical deployment inputs.
* Fixed task-space policy export to serialize the configured Cartesian action
  scales.
* Fixed Newton DisplayPort gripper resets to preserve mimic-joint behavior
  across physics-backend joint orders.
* Fixed Newton 1.5 DisplayPort asset loading by applying point-SDF properties
  only to meshes authored for SDF collision.
