Added
^^^^^

* Added configurable mesh SDF resolution to
  :class:`~isaaclab_newton.physics.HydroelasticSDFCfg`.
* Added independent mesh SDF cooking for Newton hard point contacts through
  :attr:`~isaaclab_newton.physics.NewtonCollisionPipelineCfg.mesh_sdf_max_resolution`.
* Added shape-path selection for mesh SDF cooking and configurable preservation
  of concave mesh colliders.
