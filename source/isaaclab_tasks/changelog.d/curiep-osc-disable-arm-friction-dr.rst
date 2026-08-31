Added
^^^^^

* Added an explicit ``FlangePose6D-ArmFrictionDR`` calibrated OSC task for
  reproducing policies trained with arm joint-friction randomization while
  keeping arm PD-gain randomization disabled.

Fixed
^^^^^

* Disabled arm joint-friction randomization for the Newton DisplayPort OSC
  task family (:func:`~isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.ik_newton_env_cfg._configure_osc_control`).
  The randomization was left active despite OSC's effort control having no
  friction compensation term, contradicting the validated PhysX task, which
  ships without arm actuator/friction randomization because the real Flexiv
  controller does not need it.
