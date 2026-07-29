Added
^^^^^

* Added distinct Newton point-SDF and hydroelastic training support for the Flexiv
  Rizon 4s gear and DisplayPort insertion tasks.
* Added Newton inverse-kinematics task variants and DisplayPort insertion
  success metrics.

Fixed
^^^^^

* Fixed Newton point-SDF presets to avoid allocating hydroelastic contact reducers and
  sized the task collision-pair buffers for 256 environments per GPU.
* Fixed gear assembly reset and reward targets to use the selected gear shaft
  and a physical gripper grasp.
* Fixed Newton Flexiv gear insertion accumulating gravity-induced arm drift by
  enabling actuator gravity compensation and backend-specific arm gains.
* Fixed Newton Flexiv DisplayPort insertion accumulating gravity-induced arm
  drift by applying passive body gravity compensation while preserving gravity
  on the plug and preserving the PhysX configuration.
