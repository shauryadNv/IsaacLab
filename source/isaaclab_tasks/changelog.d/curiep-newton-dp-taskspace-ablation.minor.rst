Added
^^^^^

* Added calibrated DisplayPort task-space variants with flange-origin or
  150 mm TCP pose observations, Newton inverse kinematics, and compliant
  operational-space control.
* Added task-space search diagnostics and terminal success metrics grouped by
  initial XY offset.
* Added point-SDF contact-gap presets and a PhysX preset using the cleaned
  DisplayPort socket asset.

Fixed
^^^^^

* Fixed DisplayPort point-SDF gap presets falling back to the PhysX backend.
* Fixed unstable Newton operational-space task defaults by reducing relative
  flange commands, enabling inertia decoupling with critical damping, removing
  competing nullspace motion, and adding an isolated action-scale ablation.
