Fixed
^^^^^

* Fixed the Newton DisplayPort OSC PhysX-profile task
  (:class:`~isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.ik_newton_env_cfg.Rizon4sGravDisplayportInsertionDomainRandomizedNewtonOSCPhysXProfileEnvCfg`)
  matching PhysX's raw joint-damping constants, which left the rotational axes
  underdamped and collapsed insertion success under Newton's OSC solver. The
  task now keeps Newton's inertia-decoupled, critically-damped OSC gains and
  only matches the PhysX task's action scale and clip.
