Added
^^^^^

* Added compliant-ALM, rigid-body contact, and joint tuning parameters to
  :class:`~isaaclab_newton.physics.VBDSolverCfg`.
* Added rigid-contact matching configuration to
  :class:`~isaaclab_newton.physics.NewtonCollisionPipelineCfg`.

Fixed
^^^^^

* Initialized VBD contact-history buffers before CUDA graph capture.
