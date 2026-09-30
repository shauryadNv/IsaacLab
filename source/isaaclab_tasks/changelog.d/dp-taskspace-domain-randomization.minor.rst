Added
^^^^^

* Added toggleable domain randomization to the task-space DisplayPort cable-insertion environment, configured through
  ``env.dr`` on :class:`~isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.Rizon4sTaskSpaceDisplayportInsertionEnvCfg`
  and inherited by the ``-TaskSpace-ROS-Inference`` task. Knobs cover operational-space controller gains, joint armature
  and friction, gripper-finger and mating friction, plug mass, grasp position and orientation, socket pose, observation
  noise, action noise and latency, and an external force on the plug. Every knob is off by default, so the
  unrandomized environment is unchanged.
* Added :class:`~isaaclab_tasks.contrib.deploy.mdp.SuccessDifficultyScheduler`, a success-driven automatic
  domain-randomization curriculum. A global difficulty level advances when the smoothed episode success rate clears a
  threshold and a minimum number of episodes' worth of steps have elapsed (5 by default), and each enabled
  randomization range is interpolated from its easy endpoint to its hard endpoint by that level.
* Added :class:`~isaaclab_tasks.contrib.deploy.mdp.randomize_osc_task_gains`, which scales operational-space controller
  task gains per environment, uniformly or log-uniformly. The task-space environment zeroes the arm joint PD so the
  controller can drive the joints with pure torque, which makes
  :class:`~isaaclab.envs.mdp.events.randomize_actuator_gains` a no-op there.
* Added :class:`~isaaclab_tasks.contrib.deploy.mdp.AdrRigidBodyMaterial`, a wrapper around
  :class:`~isaaclab.envs.mdp.events.randomize_rigid_body_material` that rebuilds its delegate when the sampling range
  changes. The stock term caches its PhysX material buckets in its constructor, so a curriculum that widens the range
  would otherwise never reach the simulator.
* Added :class:`~isaaclab_tasks.contrib.deploy.mdp.NoisyDelayedOperationalSpaceControllerActionCfg`, an operational-space
  action term with per-episode action bias, per-step action noise, and per-environment command latency.
* Added :class:`~isaaclab_tasks.contrib.deploy.mdp.AdrResetRootStateUniform`, a
  :class:`~isaaclab.envs.mdp.events.reset_root_state_uniform` whose pose and velocity ranges may change at runtime. The
  stock term converts its ranges to tensors in its constructor and ignores the range it is later called with.
* Added a ``rot_randomization_range`` parameter to
  :class:`~isaaclab_tasks.contrib.deploy.mdp.set_robot_to_object_grasp_pose` for per-reset randomization of the grasp
  orientation.
* Added optional payload gravity compensation to
  :class:`~isaaclab_tasks.contrib.deploy.mdp.DeployOperationalSpaceControllerActionCfg`
  (``payload_gravity_compensation``, ``payload_asset_name``, ``payload_mass_scale``). It adds the joint torques that
  hold up a grasped rigid object's weight at its center of mass. The task-space arm is gravity-free but the plug is
  not, and with ``pose_rel`` targets re-based on the measured pose each step the gripper otherwise sinks under zero
  actions (about 29 mm in 5 s), unlike the real robot's position servo. The task-space environment sets
  ``payload_asset_name="dp_plug"``; compensation is off by default.
* Added a ``spawned_at_goal`` mask to :class:`~isaaclab_tasks.contrib.deploy.mdp.reset_plug_at_goal_curriculum`. The
  success-driven curriculum uses it to score only episodes that began at the approach pose, since episodes that begin
  partially inserted would otherwise inflate the success rate it advances on.

Fixed
^^^^^

* Fixed :class:`~isaaclab_tasks.contrib.deploy.mdp.reset_plug_at_goal_curriculum` ignoring ``at_goal_prob`` and
  ``at_goal_prob_final`` changes made after construction, which left a curriculum-driven at-goal schedule stuck at its
  initial value.
* Fixed resumed :class:`~isaaclab_tasks.contrib.deploy.cable_insertion.DisplayportInsertionEnv` runs restarting the
  at-goal spawn anneal from its initial probability. ``load_training_state`` now restores ``common_step_counter`` to the
  exact cumulative rollout boundary stored with the checkpoint, and
  :class:`~isaaclab_tasks.contrib.deploy.mdp.SuccessDifficultyScheduler` restarts its level-change spacing at that
  step instead of at 0.
* Fixed DisplayPort randomization preserving disabled socket pose components, sampling one grasp target per reset,
  keeping rotation observations on SO(3), separating dry joint friction from viscous damping, and enforcing consistent
  sampled contact friction.

Changed
^^^^^^^

* Changed the ``osc_stiffness`` and ``osc_damping_ratio`` domain-randomization ranges to softer-only (stiffness scale
  0.5-1.0) and more-damped-only (damping-ratio scale 1.0-1.5). With the nominal task-space gains at the edge of stable
  plug-socket contact, raising translational stiffness by 10% or lowering damping by 10% made the articulation blow up
  during insertion in some environments; softer or more damped gains do not.

* Changed :class:`~isaaclab_tasks.contrib.deploy.cable_insertion.DisplayportInsertionEnv` to track a sticky
  per-episode ``episode_succeeded`` flag and to expand ``env.dr`` during construction. Randomization is expanded there
  rather than in ``__post_init__`` because Isaac Lab applies ``env.*`` command-line overrides after the configuration
  object is built.
* Changed :class:`~isaaclab_tasks.contrib.deploy.cable_insertion.DisplayportInsertionEnv` to persist the
  domain-randomization curriculum in a checkpoint-specific sidecar and reapply its restored ranges before the first
  resumed rollout. RSL-RL checkpoints carry only the networks, optimizer and iteration count, so a resumed run would
  otherwise use an unrelated curriculum level or retain the initial randomization ranges.
* Changed :class:`~isaaclab_tasks.contrib.deploy.mdp.SuccessDifficultyScheduler` to weight its smoothed success rate by
  completed episode count and include successes reached on the terminal physics step.
