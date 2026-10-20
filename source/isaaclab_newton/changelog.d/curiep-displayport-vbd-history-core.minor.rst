Added
^^^^^

* Added rigid-contact history and collision matching controls for
  :class:`~isaaclab_newton.physics.VBDSolverCfg` experiments.

Fixed
^^^^^

* Initialized VBD contact-history buffers before solver construction and kept
  reduced articulation state synchronized after each VBD solve.
