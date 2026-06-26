Added
^^^^^

* Added Newton backend support to the contrib gear-assembly task
  (``IsaacContrib-Deploy-GearAssembly-Rizon4s-Grav``), including a ``default`` MuJoCo-contacts
  preset and a ``newton_hydroelastic`` SDF-contacts preset, and gripper actuator tuning
  (damping and armature) for stable grasping on Newton.
* Added package-local Newton SDF gear assets for the ``newton_hydroelastic`` preset while keeping
  the default and PhysX presets on the original Factory USD assets.
* Changed gear-assembly reset placement so the non-selected gears stay inserted at their
  shaft rest height in the randomized base frame, while the selected gear is lifted from its
  shaft to be grasped for insertion training.
* Added ``IsaacContrib-Deploy-GearAssembly-Rizon4s-Grav-Newton-IK``, a variant that drives the arm
  with a Newton inverse-kinematics task-space end-effector pose action instead of joint-space
  control (requires a Newton preset).

Fixed
^^^^^

* Fixed Rizon 4s gear-assembly reset stability under Newton hydroelastic contacts by
  using contact-compatible gripper reset widths and hub targets while keeping non-selected
  gears aligned with their measured base shaft centers.
* Fixed Rizon 4s Newton gear-assembly grasp stability with contact-compatible gripper
  actuator tuning and hub-centered grasp reset targets.
* Fixed Rizon 4s gear-assembly insertion rewards so the selected gear is tracked to
  the selected shaft on the gear base instead of the gear-base origin.
