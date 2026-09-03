Added
^^^^^

* Added nominal and calibrated DisplayPort task-space variants with
  flange-origin or 150 mm TCP pose observations, Newton inverse kinematics,
  and compliant operational-space control.
* Added task-space search diagnostics and terminal success metrics grouped by
  initial XY offset.
* Added point-SDF contact-gap presets and a PhysX preset using the cleaned
  DisplayPort socket asset.
* Added explicit full-range Newton IK and operational-space action variants
  for comparison with PhysX task-space policies.
* Added nominal and calibrated Newton IK and operational-space variants that
  observe and command a matching TCP frame 150 mm from the Rizon 4s flange.
* Added a calibrated task-impedance variant for comparison with
  inertia-decoupled operational-space control.

Fixed
^^^^^

* Fixed DisplayPort point-SDF gap presets falling back to the PhysX backend.
* Fixed unstable Newton operational-space task defaults by reducing relative
  flange commands, enabling inertia decoupling with critical damping, removing
  competing nullspace motion, and adding an isolated action-scale ablation.
* Fixed DisplayPort training accepting plugs that tunneled through the socket
  by terminating trajectories beyond the mate-plane tolerance.
* Fixed Newton DisplayPort operational-space control bypassing configured arm
  effort limits by using explicit zero-gain actuators for torque commands.
