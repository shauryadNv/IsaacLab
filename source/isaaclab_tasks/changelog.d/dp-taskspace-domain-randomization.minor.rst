Added
^^^^^

* Added toggleable domain randomization to the task-space DisplayPort cable-insertion environment, configured through
  ``env.dr`` on :class:`~isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.Rizon4sTaskSpaceDisplayportInsertionEnvCfg`
  and inherited by the ``-TaskSpace-ROS-Inference`` task. Knobs cover operational-space controller gains, joint armature
  and friction, gripper-finger and mating friction, plug mass, grasp position and orientation, socket pose, observation
  noise, action noise and latency, and an external wrench on the plug. Every knob is off by default, so the
  unrandomized environment is unchanged.
* Added :class:`~isaaclab_tasks.contrib.deploy.mdp.SuccessDifficultyScheduler`, a success-driven automatic
  domain-randomization curriculum. A global difficulty level advances when the smoothed episode success rate clears a
  threshold and a minimum number of steps have elapsed, and each enabled randomization range is interpolated from its
  easy endpoint to its hard endpoint by that level.
* Added :class:`~isaaclab_tasks.contrib.deploy.mdp.randomize_osc_task_gains`, which scales operational-space controller
  task gains per environment. The task-space environment zeroes the arm joint PD so the controller can drive the joints
  with pure torque, which makes :class:`~isaaclab.envs.mdp.events.randomize_actuator_gains` a no-op there.
* Added :class:`~isaaclab_tasks.contrib.deploy.mdp.AdrRigidBodyMaterial`, a wrapper around
  :class:`~isaaclab.envs.mdp.events.randomize_rigid_body_material` that rebuilds its delegate when the sampling range
  changes. The stock term caches its PhysX material buckets in its constructor, so a curriculum that widens the range
  would otherwise never reach the simulator.
* Added :class:`~isaaclab_tasks.contrib.deploy.mdp.NoisyDelayedOperationalSpaceControllerActionCfg`, an operational-space
  action term with per-episode action bias, per-step action noise, and per-environment command latency.
* Added a ``rot_randomization_range`` parameter to
  :class:`~isaaclab_tasks.contrib.deploy.mdp.set_robot_to_object_grasp_pose` for per-reset randomization of the grasp
  orientation.

Changed
^^^^^^^

* Changed :class:`~isaaclab_tasks.contrib.deploy.cable_insertion.DisplayportInsertionEnv` to track a sticky
  per-episode ``episode_succeeded`` flag and to expand ``env.dr`` during construction. Randomization is expanded there
  rather than in ``__post_init__`` because Isaac Lab applies ``env.*`` command-line overrides after the configuration
  object is built.
* Changed :class:`~isaaclab_tasks.contrib.deploy.cable_insertion.DisplayportInsertionEnv` to implement
  ``load_training_state``, which restores the domain-randomization curriculum level from an ``adr_state.json`` sidecar
  written beside the run's checkpoints. RSL-RL checkpoints carry only the networks, optimizer and iteration count, so a
  resumed run would otherwise restart the curriculum at level 0 while keeping hard-trained policy weights.
