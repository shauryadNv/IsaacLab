.. _walkthrough_dp_insertion:

Training a DisplayPort Cable Insertion Policy and ROS Deployment
================================================================

This tutorial walks you through how to train a DisplayPort plug insertion reinforcement learning (RL) policy that transfers from simulation to a real Flexiv robot. The workflow consists of two main stages:

1. **Simulation Training in Isaac Lab**: Train the policy in a high-fidelity physics simulation with domain randomization
2. **LEAPP Export and Real Robot Deployment**: Export the trained policy with LEAPP, then deploy on hardware with Isaac ROS

This walkthrough covers the key principles and best practices for sim-to-real transfer using Isaac Lab.

**Supported Robot:**

- **Flexiv Rizon 4s**: 7-DOF collaborative robot arm with Grav parallel gripper

This environment has been successfully deployed and tested on a real Flexiv Rizon 4s robot without an IsaacLab dependency.

**Task Details:**

The DisplayPort insertion policy operates as follows:

1. **Initial State**: The policy assumes the DisplayPort plug is already grasped by the gripper at the start of the episode
2. **Input Observations**: The policy receives the current state of the robot together with the pose of the socket insertion point (position and orientation) from a separate perception pipeline. The robot state is the arm joint angles in the joint-space variants, or the flange / end-effector pose in the task-space variants
3. **Policy Output**: The policy outputs an incremental command each step — delta joint positions (incremental changes to arm joint angles) in the joint-space variants, or a 6-DoF Cartesian end-effector pose delta in the task-space variants
4. **Task Goal**: Insert the DisplayPort plug into a fixed socket until the mate point aligns within the success threshold

Both control spaces are supported end-to-end. **Task space is the recommended route for real-robot deployment**;
see :ref:`choosing-control-space` for the trade-offs and for the extra robot-calibration step joint space requires.

**Scope of This Tutorial:**

This tutorial covers **training and LEAPP export** in Isaac Lab. For the complete on-robot workflow (vision pipeline, robot interface, ROS inference), refer to the `Isaac ROS Documentation <https://nvidia-isaac-ros.github.io/reference_workflows/isaac_for_manipulation/packages/isaac_ros_manipulation_dnn_policy/index.html>`_ after exporting your policy.

**Code Layout:**

The task follows the same structure as the gear assembly deploy environments:

- ``isaaclab_tasks/contrib/deploy/cable_insertion/displayport_insertion_env_cfg.py`` — shared task MDP (scene, assets, observations, rewards)
- ``isaaclab_tasks/contrib/deploy/cable_insertion/insertion_env.py`` — environment class that logs insertion success metrics during training
- ``isaaclab_tasks/contrib/deploy/cable_insertion/config/displayport_rizon_4s/`` — Flexiv Rizon 4s + Grav robot-specific overrides and gym registrations
- ``isaaclab_tasks/contrib/deploy/cable_insertion/config/displayport_rizon_4s/joint_pos_env_cfg.py`` — joint-space (relative joint position) environment
- ``isaaclab_tasks/contrib/deploy/cable_insertion/config/displayport_rizon_4s/task_space_env_cfg.py`` — task-space (operational space control) environment
- ``isaaclab_tasks/contrib/deploy/cable_insertion/config/displayport_rizon_4s/task_space_newton_env_cfg.py`` — Newton point-SDF task-space environment
- ``isaaclab_tasks/contrib/deploy/cable_insertion/config/displayport_rizon_4s/task_space_newton_ros_inference_env_cfg.py`` — matching Newton deployment contract
- ``scripts/reinforcement_learning/train.py`` — unified trainer (pass ``--rl_library rsl_rl``)
- ``scripts/reinforcement_learning/deploy/play_displayport_insertion.py`` — DisplayPort raw-checkpoint inference, joint-space LEAPP validation, and CSV logging
- ``scripts/reinforcement_learning/leapp/rsl_rl/export_displayport_insertion.py`` — DisplayPort LEAPP exporter (task-space I/O contract)

Overview
--------

Successful sim-to-real transfer requires addressing three fundamental aspects:

1. **Input Consistency**: Ensuring the observations your policy receives in simulation match those available on the real robot
2. **System Response Consistency**: Ensuring the robot and environment respond to actions in simulation the same way they do in reality
3. **Output Consistency**: Ensuring any post-processing applied to policy outputs in Isaac Lab is also applied during real-world inference

When all three aspects are properly addressed, policies trained purely in simulation can achieve robust performance on real hardware without any real-world training data.

**Debugging Tip**: When your policy fails on the real robot, set up the real robot with the same initial observations as in simulation, then compare how the controller responds. This isolates whether the problem is from observation mismatch (Input Consistency) or physics/controller mismatch (System Response Consistency).

Asset Quality for Insertion Tasks
----------------------------------

For any contact-rich insertion task, **the quality of the plug and socket assets matters more than most other sim-to-real knobs**. DisplayPort insertion in particular operates at very small clearances between plug blades and the socket cavity. If the USD collision geometry, mass properties, or joint behavior are wrong, no amount of reward tuning or domain randomization will produce a policy that transfers well to hardware.

The DisplayPort plug and socket assets used by these environments have been iterated extensively and work well for training policies that transfer sim-to-real. Expect considerable upfront effort to reach this quality for a new connector or cable type.

**What to validate before training:**

1. **Static insertion pose stability**: Load the plug fully inserted into the socket at the goal pose (no robot, no gripper). The plug should remain seated without drifting, jittering, or being ejected by contact forces. Persistent separation or slow creep at the mated pose usually indicates incorrect collision meshes, rest offsets, or mass/inertia.
2. **Collision fidelity at clearance scale**: Blade-to-cavity gaps are sub-millimeter. Convex hulls or coarse meshes often produce false contacts, snagging, or penetration. SDF or carefully authored triangle meshes with tuned ``contact_offset`` / ``rest_offset`` are typically required.
3. **Engagement behavior**: Push the plug through the approach path by hand (or with scripted motion) and confirm contact feels plausible — no explosive pops, no tunneling through the socket wall, no sticky high-friction jamming unless that matches the real connector.
4. **Grasped plug stability**: With the gripper closed at the training grasp width, the plug should not spin or slip unrealistically when the arm moves. Cable mass and plug COM should be representative of the real assembly.
5. **Mate-point alignment**: Verify ``SOCKET_INSERTION_OFFSET``, ``PLUG_INSERTION_OFFSET``, and ``PLUG_GOAL_ROT`` in ``displayport_insertion_env_cfg.py`` match the intended physical mate frame. Reward and success metrics are computed from these offsets; a mismatch here looks like a perception error on the real robot.

**The robot asset matters too.** The checks above concern the plug and socket, but the arm's USD must also
represent *your* robot — particularly its kinematic parameters, which may vary unit to unit. This matters most for
joint-space training, where the policy commands joints directly. Flexiv's
`flexiv_calibration <https://github.com/flexivrobotics/flexiv_ros2/tree/release/lyrical-v1.9.3/flexiv_calibration>`__
workflow exports a calibrated robot description for a specific arm; convert the result to USD and set
``scene.robot.spawn.usd_path`` in a derived environment configuration. The Newton profile uses the
calibrated USD for the reference Rizon 4s with serial number ``063459`` by default. That asset is not a generic
calibration for every Rizon 4s. For another robot, pass its calibrated USD as a Hydra override when training or
playing a policy:

.. code-block:: text

   env.scene.robot.spawn.usd_path=/absolute/path/to/your_calibrated_robot.usd

**Practical workflow:**

1. Fix assets in isolation (drop-test or basic play env with fixed poses) before running full RL training.
2. Compare sim behavior to real hardware video at the same poses — look for drift, bounce, and penetration, not policy success rate.
3. Only after assets pass these checks, tune curriculum, rewards, and domain randomization.

.. note::

   Poor asset quality often shows up as policies that learn high training success but fail on hardware with inconsistent contact behavior, or as training that never achieves high ``Metrics/success_rate`` despite reward tuning. Fix the assets first.

Part 1: Input Consistency
--------------------------

The observations your policy receives must be consistent between simulation and reality. This means:

1. The observation space should only include information available from real sensors
2. Sensor noise and delays should be modeled appropriately

Using Real-Robot-Available Observations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Your simulation environment should only use observations that are available on the real robot and not use "privileged" information that would not be available in deployment. The critic receives additional privileged observations (plug pose and joint velocities) to improve value estimation during training, but these are not passed to the actor at deployment time.


Observation Specification
^^^^^^^^^^^^^^^^^^^^^^^^^

The DisplayPort insertion environment uses proprioceptive and exteroceptive (vision) observations. In every
variant the actor sees **the current state of the robot** plus **the socket insertion (mate) point** supplied by
perception. What differs between the two control spaces is how the robot state is represented: joint angles for
the joint-space environments, and the flange / end-effector pose for the task-space environments.

.. tab-set::

   .. tab-item:: Joint space

      .. list-table:: Joint-Space Observations (Flexiv Rizon 4s)
         :widths: 25 10 25 20
         :header-rows: 1

         * - Observation
           - Dim
           - Real-World Source
           - Noise
         * - ``joint_pos`` (arm only)
           - 7
           - Robot controller
           - None
         * - ``joint_vel`` (arm only, optional)
           - 7
           - Robot controller
           - None
         * - ``socket_pos`` (insertion mate point)
           - 3
           - Perception pipeline
           - ±10mm
         * - ``socket_quat``
           - 4
           - Perception pipeline
           - None

      **Recommended shipping configuration** (``NoJointVel`` variants): **14** policy dimensions
      (7 joint positions + 3 socket position + 4 socket quaternion).

      **Training configuration with joint velocity** (``Grav`` variants without ``NoJointVel``): **21** policy dimensions.

      .. note::

         **Within joint space, prefer the NoJointVel variants.** Policies trained with ``joint_vel`` in the actor
         observation can achieve slightly higher success rates in simulation, but we consistently observe less
         stable behavior on the real Flexiv robot (jittery motions, inconsistent contact during insertion).

         The ``NoJointVel`` configs remove ``joint_vel`` from the actor observation while keeping it in the critic
         observation group. This matches deployment setups where joint velocity is not exposed to the policy
         network but can still help the value function during training.

         This is a choice *among the joint-space variants*; the task-space environments are recommended over
         joint space generally (see :ref:`choosing-control-space`), and they do not expose joint velocity to the
         actor at all.

   .. tab-item:: Task space

      .. list-table:: Task-Space Observations (Flexiv Rizon 4s)
         :widths: 25 10 25 20
         :header-rows: 1

         * - Observation
           - Dim
           - Real-World Source
           - Noise
         * - ``eef_pos`` (TCP position)
           - 3
           - Robot controller
           - None
         * - ``eef_rot_6d`` (TCP orientation)
           - 6
           - Robot controller
           - None
         * - ``socket_kp_pos`` (insertion mate point)
           - 3
           - Perception pipeline
           - None by default
         * - ``socket_kp_rot_6d``
           - 6
           - Perception pipeline
           - None by default

      **Shipping configuration** (``TaskSpace`` variants): **18** policy dimensions
      (3 + 6 end-effector pose + 3 + 6 socket keypoint frame).

      Orientations use a **6D rotation representation** (the first two rows of the rotation matrix) rather than a
      quaternion. This is continuous over SO(3), which avoids the sign-flip discontinuity a quaternion introduces
      and is easier for the network to regress against.

      The end-effector position is reported at the **tool center point (TCP)**, i.e. the ``flange`` body offset
      along its local +Z by ``_TCP_OFFSET`` in ``task_space_env_cfg.py``. Note that the observation is taken at the
      TCP while the action is applied at the flange; see :ref:`taskspace-action-space`.

**Implementation (base class):**

.. code-block:: python

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )
        socket_pos = ObsTerm(
            func=mdp.rigid_object_pos_w,
            params={"asset_cfg": SceneEntityCfg("dp_socket"), "offset": SOCKET_INSERTION_OFFSET},
            noise=ResetSampledConstantNoiseModelCfg(
                noise_cfg=UniformNoiseCfg(n_min=-0.01, n_max=0.01, operation="add")  # ±10mm
            ),
        )
        socket_quat = ObsTerm(
            func=mdp.rigid_object_quat_w,
            params={"asset_cfg": SceneEntityCfg("dp_socket")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

**Rizon 4s overrides** (in ``config/displayport_rizon_4s/joint_pos_env_cfg.py``):

.. code-block:: python

    # Arm joints only — gripper joints are excluded from observations
    self.observations.policy.joint_pos.params["asset_cfg"].joint_names = [
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
    ]

**Task-space actor group** (in ``config/displayport_rizon_4s/task_space_env_cfg.py``):

.. code-block:: python

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations: EEF pose + socket keypoint frame (18 dims)."""

        eef_pos = ObsTerm(
            func=mdp.eef_pos_w,
            params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "flange", "offset": _TCP_OFFSET},
        )
        eef_rot_6d = ObsTerm(
            func=mdp.eef_rot_6d_w,
            params={"asset_cfg": SceneEntityCfg("robot"), "body_name": "flange"},
        )
        socket_kp_pos = ObsTerm(
            func=mdp.rigid_object_pos_w,
            params={"asset_cfg": SceneEntityCfg("dp_socket"), "offset": SOCKET_INSERTION_OFFSET},
        )
        socket_kp_rot_6d = ObsTerm(
            func=mdp.rigid_object_rot_6d_w,
            params={"asset_cfg": SceneEntityCfg("dp_socket")},
        )

In both control spaces the **critic** additionally receives privileged state that the actor never sees: joint
positions and velocities plus the plug keypoint frame (task space) or the plug pose (joint space).

**Why No Noise for Proprioceptive Observations?**

As with the gear assembly task, policies trained without noise on proprioceptive observations transfer well to the Flexiv Rizon 4s. The controller provides sufficiently accurate joint state feedback that modeling sensor noise on joint states does not improve sim-to-real transfer for this task. The same reasoning carries over to the task-space variants: the end-effector pose is derived from those same joint encoders through forward kinematics, so it inherits their accuracy and is likewise left noise-free.

The socket pose, by contrast, comes from perception and *is* the observation worth corrupting. The joint-space
environments apply ±10 mm of reset-sampled uniform noise to ``socket_pos`` by default. The existing PhysX
task-space environments disable observation noise. The Newton profile applies reset-held ±10 mm noise;
change it only together with the perception contract used for deployment.


Part 2: System Response Consistency
------------------------------------

Once your observations are consistent, ensure the simulated robot and environment respond to actions the same way the real system does. For DisplayPort insertion this involves:

1. Physics simulation parameters (friction, contact properties, plug/socket collision meshes)
2. Actuator modeling (PD controller gains, effort limits)
3. Domain randomization and curriculum

Physics Parameter Tuning
~~~~~~~~~~~~~~~~~~~~~~~~~

Accurate physics simulation is critical for contact-rich insertion. The DisplayPort plug and socket use SDF collision meshes with high solver iteration counts on the rigid bodies:

.. code-block:: python

    # From displayport_insertion_env_cfg.py — DisplayPortPlug / DisplayPortSocket
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        solver_position_iteration_count=128,
        solver_velocity_iteration_count=1,
        max_depenetration_velocity=0.5,  # plug; socket uses 5.0
    ),
    collision_props=sim_utils.CollisionPropertiesCfg(
        contact_offset=0.00001,   # plug
        rest_offset=-0.00005,
    ),

The Flexiv Rizon 4s arm uses lower solver iteration counts for performance, matching the gear assembly Flexiv configuration:

.. code-block:: python

    # From config/displayport_rizon_4s/joint_pos_env_cfg.py
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=True,
        max_depenetration_velocity=5.0,
        solver_position_iteration_count=4,
        solver_velocity_iteration_count=1,
        max_contact_impulse=1e32,
    ),
    collision_props=sim_utils.CollisionPropertiesCfg(
        contact_offset=0.005,
        rest_offset=0.0,
    ),

**Friction randomization** (in ``config/displayport_rizon_4s/joint_pos_env_cfg.py``):

.. code-block:: python

    plug_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("dp_plug", body_names=".*"),
            "static_friction_range": (0.001, 0.001),
            "dynamic_friction_range": (0.001, 0.001),
        },
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*finger.*"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
        },
    )

Low plug/socket friction reduces sticking during blade engagement. Gripper finger friction is set to match real grasp behavior.

Newton Point-SDF Profile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the dedicated Newton task-space profile when training an operational-space policy with the Newton backend.
It is additive to the existing PhysX environments and preserves the training and deployment ABI of the reference
hardware-validated configuration. A checkpoint is not bundled; train a new policy with the task id below.

.. important::

   The Newton and PhysX task-space actors are both 18-dimensional, but their observation contracts are not
   interchangeable. The Newton actor uses this exact order:

   #. ``socket_pos`` (3)
   #. ``tool_pos`` at the flange origin (3)
   #. ``tool_rot_6d`` (6)
   #. ``socket_rot_6d`` (6)

   The PhysX task-space actor instead uses TCP-first ordering. A checkpoint can therefore load under the wrong
   task id without a shape error and still receive semantically incorrect inputs. Always use a Newton task id for
   a checkpoint trained with the Newton contract.

The ``display_port_*_newton_sdf.usda`` files are small metadata overlays on the versioned DisplayPort geometry
shipped with this package. They stay next to their relative sublayers so installed-package and offline asset
resolution are deterministic; a Nucleus dependency is not required for these task-specific schemas.

The Newton actor applies reset-held socket-position noise in ``[-0.01, 0.01]`` m. For parity with the reference
training run, one scalar is sampled per environment and broadcast across XYZ. Independent per-axis noise is a
different training distribution. The privileged critic retains all 13 robot joints and has 40 inputs.

.. list-table:: Newton timing and contact settings
   :widths: 42 23 35
   :header-rows: 1

   * - Setting
     - Value
     - Effective rate / purpose
   * - Outer simulation step (``sim.dt``)
     - ``0.01`` s
     - 100 Hz environment step
   * - Newton solver substeps
     - ``20``
     - 2,000 Hz solver
   * - ``collision_decimation``
     - ``10``
     - 200 Hz collision updates
   * - Environment ``decimation``
     - ``3``
     - approximately 33.3 Hz policy
   * - Contact model / gap
     - point SDF / ``0.005`` m
     - Newton collision pipeline, not MuJoCo contacts
   * - Solver / integrator
     - Newton / ``implicitfast``
     - MJWarp Newton solve method
   * - Contact reduction / triangle-pair capacity
     - enabled / ``2**25``
     - scene-wide capacity for the 256-environment per-rank default

This is an MJWarp point-SDF configuration. VBD and hydroelastic SDF are separate experiments and must be evaluated
independently. ``max_triangle_pairs`` is scene-wide: when Newton reports an overflow, first reduce the per-rank
environment count. If memory permits, the capacity can instead be doubled from the default with
``env.sim.physics.collision_cfg.max_triangle_pairs=67108864``. Overflow can omit candidate contacts.

The policy emits a six-dimensional relative pose command at the flange origin. RSL-RL first clips each raw actor
output to ``[-1, 1]``; the OSC action then applies ``0.025`` translation and rotation scales. The action-term clip
is intentionally unset so this transform has only one clipping stage.
The OSC stiffness is ``(300, 300, 300, 30, 30, 30)`` with damping ratio ``1.0`` on every axis.
Full inertial-dynamics decoupling is enabled; partial decoupling and null-space control are disabled. Arm joint-PD
stiffness and damping are zero so OSC supplies the arm effort. Newton rigid-body gravity compensation is enabled
for the robot, while OSC gravity compensation is disabled to avoid applying it twice.

The profile also preserves the reference domain randomization and curriculum: plug/socket/finger friction values
of ``3.0`` / ``0.001`` / ``1.0``, additive arm-joint friction randomization in ``[0.0, 0.15]``, no arm PD-gain
randomization, and at-goal resets annealed from ``0.8`` to ``0.0`` over iterations 0 through 500.

Train the deployment-compatible task for 1,000 iterations. Passing ``presets=newton_sdf`` explicitly records the
selected contact profile even though it is also the Newton task's default:

.. code-block:: bash

    ./isaaclab.sh train --rl_library rsl_rl \
        --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Newton-ROS-Inference-v0 \
        --num_envs 256 \
        --seed 126 \
        --max_iterations 1000 \
        presets=newton_sdf

Evaluate a checkpoint with deterministic actor observations:

.. code-block:: bash

    ./isaaclab.sh play --rl_library rsl_rl \
        --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Newton-Play-v0 \
        --num_envs 1 \
        --checkpoint logs/rsl_rl/displayport_insertion_rizon4s_newton_osc/<run>/model_999.pt \
        --deterministic \
        --visualizer kit \
        presets=newton_sdf

Export from the matching ROS-inference task so LEAPP uses the socket-first/flange-origin observation metadata:

.. code-block:: bash

    ./isaaclab.sh -p scripts/reinforcement_learning/leapp/rsl_rl/export_displayport_insertion.py \
        --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Newton-ROS-Inference-v0 \
        --checkpoint logs/rsl_rl/displayport_insertion_rizon4s_newton_osc/<run>/model_999.pt \
        --task_space_contract \
        presets=newton_sdf


PhysX Actuator Modeling
~~~~~~~~~~~~~~~~~~~~~~~

The PhysX Rizon 4s profiles use ``ImplicitActuatorCfg`` with per-joint-group arm tuning from
``FLEXIV_RIZON4S_GRAV_GRIPPER_CFG``, plus these dedicated Grav gripper actuators:

.. code-block:: python

    # Grav gripper actuator configuration
    self.scene.robot.actuators["gripper_drive"] = ImplicitActuatorCfg(
        joint_names_expr=["finger_joint"],
        effort_limit_sim=2.0,
        velocity_limit_sim=1.0,
        stiffness=2e3,
        damping=1e1,
    )
    self.scene.robot.actuators["gripper_passive"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_knuckle_joint"],
        effort_limit_sim=1.0,
        velocity_limit_sim=1.0,
        stiffness=0.0,
        damping=0.0,
    )

The Newton profile uses the following backend-specific gripper settings:

- ``gripper_drive`` controls ``finger_joint`` with effort / velocity limits of ``200.0`` / ``2.0``, stiffness /
  damping of ``2000.0`` / ``10.0``, zero friction, and armature ``0.1``.
- ``gripper_passive`` controls both ``.*_knuckle_joint`` and ``.*_outer_finger_joint`` with effort / velocity
  limits of ``20.0`` / ``1.0``, stiffness / damping of ``2000.0`` / ``10.0``, zero friction, and armature ``0.05``.
- ``hand_hold_width`` and ``hand_close_width`` are both ``-0.1``.

These values define the Newton gripper behavior; do not substitute the PhysX gripper settings when training or
playing a Newton policy.

.. note::

   **Flexiv Rizon 4s (PhysX profiles)**: actuator-gain and joint-friction randomization is not included. The Newton
   profile retains additive uniform arm-joint friction randomization in ``[0.0, 0.15]`` and disables PD-gain
   randomization.

.. _taskspace-action-space:

Action Space Design
~~~~~~~~~~~~~~~~~~~

In both control spaces the policy commands only the arm; the gripper is not in the action space — the plug is held
at a fixed grasp width for the episode. The action is always an **incremental** (delta) command, never an absolute
target. What differs is the space that delta lives in.

.. tab-set::

   .. tab-item:: Joint space

      The policy controls the 7 arm joints using **incremental joint position control**.

      .. code-block:: python

          self.joint_action_scale = 0.025  # ±1.4 degrees per step

          self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
              asset_name="robot",
              joint_names=["joint1", "joint2", "joint3", "joint4",
                           "joint5", "joint6", "joint7"],
              scale=self.joint_action_scale,
              use_zero_offset=True,
          )

      **Action dimension:** 7 — one delta per arm joint, in radians.

   .. tab-item:: Task space

      The policy emits a **6-DoF Cartesian pose delta** which an Operational Space Controller (OSC) converts into
      joint efforts. A task-space bridge reduces sensitivity to differences in the simulated and real joint servo
      response, but transfer still depends on the robot kinematics, dynamics, controller, and frame conventions.

      .. code-block:: python

          _ACTION_SCALE = 0.025          # 25 mm and 0.025 rad per step
          _STIFFNESS = (300.0, 300.0, 300.0, 30.0, 30.0, 30.0)

          self.actions.arm_action = mdp.DeployOperationalSpaceControllerActionCfg(
              asset_name="robot",
              joint_names=_ARM_JOINTS,
              body_name="flange",
              controller_cfg=OperationalSpaceControllerCfg(
                  target_types=["pose_rel"],
                  motion_stiffness_task=_STIFFNESS,
                  ...
              ),
              position_scale=_ACTION_SCALE,
              orientation_scale=_ACTION_SCALE,
          )

      **Action dimension:** 6 — ``[dx, dy, dz, dθx, dθy, dθz]``, the first three in metres and the last three in
      radians.

      Two details matter for deployment:

      * The arm's joint PD gains are **zeroed** (``actuators[...].stiffness = 0.0``); all compliance comes from the
        task-space stiffness above, so the controller — not the joint servo — sets the contact behavior.
      * In the PhysX task-space profile, the action is applied at the **flange**, while ``eef_pos`` is observed at
        the **TCP**. The Newton profile instead observes and controls at the **flange origin**. The
        real-robot bridge must reproduce the frame contract of the selected backend.

**Action scale:** ``0.025`` in both cases — read as radians per joint per step in joint space, and as metres /
radians per step in task space.

**Control frequency:** the PhysX profiles use ``sim.dt = 1/240`` s with ``decimation = 8`` (30 Hz policy rate).
The Newton profile uses a 100 Hz outer step, 20 solver substeps, and ``decimation = 3`` (approximately 33.3 Hz);
collision detection runs every 10 substeps (200 Hz).

Domain Randomization Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Socket pose randomization** perturbs the fixed socket to cover perception and mounting variation:

.. code-block:: python

    randomize_socket_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.01, 0.01],                        # ±1 cm
                "y": [-0.01, 0.01],                        # ±1 cm
                "z": [-0.02, 0.02],                        # ±2 cm
                "roll": [-math.radians(2.0), math.radians(2.0)],
                "pitch": [-math.radians(2.0), math.radians(2.0)],
                "yaw": [-math.radians(2.0), math.radians(2.0)],
            },
            "asset_cfg": SceneEntityCfg("dp_socket"),
        },
    )

**Plug reset curriculum** starts episodes with the plug near the goal pose and anneals toward farther approach poses:

.. code-block:: python

    reset_plug_curriculum = EventTerm(
        func=mdp.reset_plug_at_goal_curriculum,
        mode="reset",
        params={
            "at_goal_prob": 0.8,
            "at_goal_prob_final": 0.0,
            "anneal_start_iter": 0.0,
            "anneal_end_iter": 500.0,
            "num_steps_per_env": 512,
            "insertion_axis": [1.0, 0.0, 0.0],
            "at_goal_depth_range": [0.0, 0.015],      # 0–15 mm engaged
            "approach_depth_range": [0.02, 0.06],     # 20–60 mm approach
            "normal_pose_range": {
                "x": [-0.02, 0.02],
                "y": [-0.02, 0.02],
                "z": [0.0, 0.0],
            },
        },
    )

At the start of training, 80% of resets place the plug near the inserted pose; this probability linearly anneals to 0% over 500 training iterations, forcing the policy to learn full approach and insertion.

**Initial robot pose** is set via inverse kinematics to a grasp pose on the plug at each reset.
``grasp_offset`` / ``end_effector_body_name`` / ``num_arm_joints`` are applied in
``Rizon4sGravDisplayportInsertionEnvCfg.__post_init__``:

.. code-block:: python

    set_robot_to_grasp_pose = EventTerm(
        func=mdp.set_robot_to_object_grasp_pose,
        mode="reset",
        params={
            "target_object_name": "dp_plug",
            "grasp_offset": [0.0025, 0.0, -0.1875],  # plug local frame [m]
            "end_effector_body_name": "flange",
            "num_arm_joints": 7,
        },
    )

Reward Shaping
~~~~~~~~~~~~~~

The environment uses keypoint-based rewards that measure alignment between the plug and socket insertion mate points. Reward terms are defined in ``displayport_insertion_env_cfg.py``:

- **Keypoint tracking** (``plug_socket_keypoint_tracking``): Penalizes L2 keypoint distance between plug and socket mate frames
- **Exponential keypoint tracking** (``plug_socket_keypoint_tracking_exp``): Dense exponential reward for fine alignment
- **Action rate** (``action_rate_l2``): Penalizes large action changes for smooth motions

The Rizon 4s config sets the linear and exponential keypoint reward weights to a **1:1 ratio**:

.. code-block:: python

    self.rewards.plug_socket_keypoint_tracking_exp.weight = abs(
        self.rewards.plug_socket_keypoint_tracking.weight
    )

Terminations
~~~~~~~~~~~~

In addition to the episode timeout, the Rizon 4s config terminates early when:

- **Plug dropped**: End-effector moves more than 15 cm away from the plug grasp point
- **Plug orientation exceeded**: Roll or pitch deviation exceeds 15° relative to the grasp frame

Training Metrics
~~~~~~~~~~~~~~~~

Unlike gear assembly, this task uses a custom environment class (``DisplayportInsertionEnv``) to log insertion metrics to TensorBoard without changing the MDP:

- ``Metrics/success_rate`` — fraction of environments within the 3 mm mate-point threshold
- ``Metrics/plug_socket_pos_error_m`` — mean mate-point distance
- ``Metrics/plug_socket_keypoint_dist_m`` — mean keypoint distance
- ``Metrics/terminal_success_rate`` — success rate at episode reset


Tuning Hyperparameters for Better Performance
----------------------------------------------

After asset quality and physics look correct, the following hyperparameters are the main levers for improving training speed, final success rate, and sim-to-real robustness. Defaults below are the shipped Flexiv Rizon 4s values; adjust one group at a time and monitor ``Metrics/terminal_success_rate`` in TensorBoard.

Reward Weights
~~~~~~~~~~~~~~

Defined in ``displayport_insertion_env_cfg.py``; the Rizon 4s config overrides the exponential weight in ``joint_pos_env_cfg.py``.

.. list-table:: Reward hyperparameters
   :widths: 35 20 45
   :header-rows: 1

   * - Parameter
     - Default
     - Effect
   * - ``plug_socket_keypoint_tracking.weight``
     - ``-1.5``
     - Linear penalty on keypoint distance. More negative → stronger pull toward alignment.
   * - ``plug_socket_keypoint_tracking_exp.weight``
     - ``1.5`` (matched to linear)
     - Exponential bonus near the goal. Increase relative to linear for sharper fine-insertion behavior; decrease if policy is brittle or stalls short of full insertion.
   * - ``kp_exp_coeffs``
     - ``[(50, 0.0001), (300, 0.0001), (600, 0.0001), (2000, 0.0001)]``
     - Per-keypoint exponential scales. Higher first values tighten the reward basin around the goal.
   * - ``keypoint_scale``
     - ``0.15``
     - Spatial extent of keypoint offsets. Affects how rotation errors contribute relative to translation.
   * - ``action_rate.weight``
     - ``-5e-6``
     - Smoothness penalty. More negative → slower, smoother motions; too strong can prevent final insertion force.

**1:1 linear-to-exponential weighting** (current shipping default):

.. code-block:: python

    self.rewards.plug_socket_keypoint_tracking_exp.weight = abs(
        self.rewards.plug_socket_keypoint_tracking.weight
    )

If the policy approaches but does not fully seat the plug, try increasing the exponential weight or tightening ``kp_exp_coeffs``. If it rushes and bounces off the socket, increase ``action_rate`` magnitude or reduce the exponential weight.

Reset Curriculum
~~~~~~~~~~~~~~~~

Defined in ``config/displayport_rizon_4s/joint_pos_env_cfg.py`` → ``reset_plug_curriculum``.

.. list-table:: Curriculum hyperparameters
   :widths: 35 20 45
   :header-rows: 1

   * - Parameter
     - Default
     - Effect
   * - ``at_goal_prob`` / ``at_goal_prob_final``
     - ``0.8`` → ``0.0``
     - Fraction of resets with plug near full insertion. Higher start values make early learning easier; anneal to zero for full approach behavior.
   * - ``anneal_end_iter``
     - ``500``
     - Training iterations over which at-goal probability anneals. Extend (e.g. 800–1000) if success rate drops when curriculum gets harder; shorten if training is too slow to reach approach poses.
   * - ``at_goal_depth_range``
     - ``[0.0, 0.015]`` m
     - How deep the plug starts when sampled "at goal" (0–15 mm engaged). Narrow for fine final-insertion practice; widen slightly if the policy never sees near-mated contacts.
   * - ``approach_depth_range``
     - ``[0.02, 0.06]`` m
     - Standoff distance when not at goal (20–60 mm). Increase upper bound for harder long-range approach; decrease if the policy struggles to reach the socket mouth.
   * - ``normal_pose_range``
     - ±2 cm lateral
     - Lateral misalignment when not at goal. Widen for more robustness to perception error; narrow if training fails to converge.

If ``Metrics/success_rate`` is high early but collapses after iteration ~500, the curriculum may be annealing too aggressively — extend ``anneal_end_iter`` or raise ``at_goal_prob_final`` temporarily.

Domain Randomization and Observations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Randomization hyperparameters
   :widths: 35 20 45
   :header-rows: 1

   * - Parameter
     - Default
     - Effect
   * - ``randomize_socket_pose`` ranges
     - ±1 cm XY, ±2 cm Z, ±2°
     - Socket pose DR. Widen to match real perception/mount error; narrow if the policy cannot learn a baseline insertion.
   * - ``socket_pos`` observation noise
     - ±10 mm
     - Perception noise on mate point. Increase for more robust real-world pose error; decrease if sim policy is too conservative.
   * - Plug/socket friction (startup)
     - ``0.001``
     - Low friction reduces unrealistic jamming. Tune only after visual sim-vs-real comparison — wrong friction can dominate insertion feel.
   * - Gripper finger friction
     - ``0.75``
     - Affects grasp stability during insertion forces.

Actions and Grasp
~~~~~~~~~~~~~~~~~~~

.. list-table:: Action / grasp hyperparameters
   :widths: 35 20 45
   :header-rows: 1

   * - Parameter
     - Default
     - Effect
   * - ``joint_action_scale``
     - ``0.025``
     - Max joint delta per step (~±1.4°). Increase if real robot stiction prevents reaching targets; decrease for finer final alignment.
   * - ``grasp_offset``
     - ``[0.0025, 0.0, -0.1875]`` m
     - EE-to-plug transform for IK reset. Wrong values cause dropped-plug terminations or misaligned approach.
   * - ``hand_hold_width`` / ``hand_close_width``
     - ``-0.05`` / ``-0.155`` rad
     - Grav finger_joint grasp command. Adjust if plug slips or is over-compressed during insertion.

Terminations and Success Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Termination / metric hyperparameters
   :widths: 35 20 45
   :header-rows: 1

   * - Parameter
     - Default
     - Effect
   * - ``success_pos_threshold``
     - ``3`` mm
     - Mate-point distance counted as success in ``Metrics/success_rate``. Tighten to match real acceptance criteria.
   * - ``plug_dropped`` distance threshold
     - ``15`` cm
     - Early reset if EE leaves plug. Tighten to discourage release; loosen if false positives during large motions.
   * - Orientation thresholds (roll/pitch)
     - ``15°``
     - Reset if plug tilts excessively relative to grasp frame.

RL Algorithm (PPO)
~~~~~~~~~~~~~~~~~~

Defined in ``config/displayport_rizon_4s/agents/rsl_rl_ppo_cfg.py``.

.. list-table:: PPO hyperparameters
   :widths: 35 20 45
   :header-rows: 1

   * - Parameter
     - Default
     - Effect
   * - ``max_iterations``
     - ``1500`` (PhysX); ``1000`` (Newton reference)
     - The reference Newton policy is the final ``model_999.pt`` checkpoint.
   * - ``num_steps_per_env``
     - ``512``
     - Rollout length per iteration. Affects curriculum annealing rate (tied to ``anneal_end_iter``).
   * - ``learning_rate``
     - ``5e-4``
     - PPO learning rate. Reduce if training is unstable; increase if learning is very slow.
   * - ``desired_kl``
     - ``0.008``
     - Target KL for adaptive LR schedule.
   * - ``init_noise_std``
     - ``1.0``
     - Exploration noise. Lower for fine-tuning a near-working policy.

**Suggested tuning order:** (1) confirm asset/physics quality, (2) curriculum depth and anneal schedule, (3) linear vs exponential reward balance, (4) socket pose DR and observation noise, (5) action scale, (6) PPO training length.


Part 3: Training the Policy in Isaac Lab
-----------------------------------------

Registered Gym Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Train on the** ``-ROS-Inference-v0`` **ids.** They are the environments used to produce the policies deployed on
hardware: in addition to the plain training configuration, they declare the observation/action metadata Isaac
Manipulator and LEAPP consume (``obs_order``, ``policy_action_space``, ``observation_space``, ``action_space``,
``action_scale``) and pin the plug, socket and seed joint pose to the real deployment station. Training on them
means the environment you trained in and the contract you export are the same thing, so a checkpoint can be
exported and deployed with Isaac ROS without swapping configurations.

.. list-table:: Flexiv Rizon 4s DisplayPort Insertion Environments
   :widths: 55 45
   :header-rows: 1

   * - Environment ID
     - Purpose
   * - ``Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Newton-ROS-Inference-v0``
     - **Newton task-space training and deployment — recommended Newton default.**
   * - ``Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Newton-v0``
     - Newton task-space training without deployment metadata
   * - ``Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Newton-Play-v0``
     - Newton task-space evaluation with observation corruption disabled
   * - ``Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-ROS-Inference-v0``
     - **Task-space training and deployment — recommended.** Carries the ROS / LEAPP export contract.
   * - ``Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-v0``
     - Task-space training without deployment metadata (ablations, experiments)
   * - ``Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Play-v0``
     - Task-space evaluation / visualization (observation corruption disabled)
   * - ``Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-ROS-Inference-v0``
     - **Joint-space training and deployment.** Carries the ROS / LEAPP export contract (14-dim actor obs).
   * - ``Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-v0``
     - Joint-space training without deployment metadata
   * - ``Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-Play-v0``
     - Evaluation / visualization (observation corruption disabled)
   * - ``Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-ROS-Inference-v0``
     - Joint-space ROS inference with joint velocity in actor obs (21-dim)
   * - ``Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-v0``
     - Training with joint velocity in actor obs (21-dim)
   * - ``Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-Play-v0``
     - Evaluation with joint velocity in actor obs

The plain training ids remain useful when you are experimenting and do not intend to deploy that particular
checkpoint — for example when sweeping rewards or randomization. If you later want to ship such a checkpoint, export
it against the matching ``-ROS-Inference-v0`` id so the traced layout matches deployment.

.. note::

   Training and play use the **unified RL entrypoints**. You must pass ``--rl_library rsl_rl``.
   The older path ``scripts/reinforcement_learning/rsl_rl/train.py`` no longer exists; use
   ``./isaaclab.sh train ...`` or ``./isaaclab.sh -p scripts/reinforcement_learning/train.py --rl_library rsl_rl ...``.
   DisplayPort-specific inference lives under ``scripts/reinforcement_learning/deploy/``
   (not the removed ``scripts/reinforcement_learning/rsl_rl/`` directory).

Step 1: Visualize the Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Launch training with a small number of environments and visualization enabled to verify the setup. Pick the tab
for the control space you intend to deploy — task space is the recommended choice for real-robot transfer
(see :ref:`choosing-control-space`):

.. tab-set::

   .. tab-item:: Joint space

      .. code-block:: bash

          ./isaaclab.sh train --rl_library rsl_rl \
              --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-ROS-Inference-v0 \
              --num_envs 4 \
              --max_iterations 100 \
              --visualizer kit

   .. tab-item:: Task space — PhysX

      .. code-block:: bash

          ./isaaclab.sh train --rl_library rsl_rl \
              --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-ROS-Inference-v0 \
              --num_envs 4 \
              --max_iterations 100 \
              --visualizer kit

   .. tab-item:: Task space — Newton

      .. code-block:: bash

          ./isaaclab.sh train --rl_library rsl_rl \
              --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Newton-ROS-Inference-v0 \
              --num_envs 4 \
              --max_iterations 100 \
              --visualizer kit \
              presets=newton_sdf

Equivalent form, invoking the unified trainer directly (substitute any task id above):

.. code-block:: bash

    ./isaaclab.sh -p scripts/reinforcement_learning/train.py --rl_library rsl_rl \
        --task <TASK_ID> \
        --num_envs 4 \
        --max_iterations 100 \
        --visualizer kit

**What to Expect:**

In early training, the robot moves the grasped plug toward the socket but will not insert reliably yet. Verify that:

- The plug is grasped at reset and held throughout the episode
- The socket pose randomization and plug curriculum produce varied starting configurations
- Contact between plug blades and socket looks physically plausible
- With the plug placed in the fully inserted pose (no policy), it stays seated without drift or instability

Stop training (Ctrl+C) once the environment looks correct, then proceed to full-scale training.

Step 2: Full-Scale Training with Video Recording
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Launch full training without an interactive visualizer and with video recording. Training is headless by default
when ``--visualizer`` is omitted; there is no ``--headless`` argument on the unified entrypoint.

.. tab-set::

   .. tab-item:: Joint space

      .. code-block:: bash

          ./isaaclab.sh train --rl_library rsl_rl \
              --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-ROS-Inference-v0 \
              --num_envs 256 \
              --video --video_length 200 --video_interval 76800

   .. tab-item:: Task space — PhysX

      .. code-block:: bash

          ./isaaclab.sh train --rl_library rsl_rl \
              --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-ROS-Inference-v0 \
              --num_envs 256 \
              --video --video_length 200 --video_interval 76800

   .. tab-item:: Task space — Newton

      .. code-block:: bash

          ./isaaclab.sh train --rl_library rsl_rl \
              --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Newton-ROS-Inference-v0 \
              --num_envs 256 \
              --seed 126 \
              --max_iterations 1000 \
              --video --video_length 222 --video_interval 76800 \
              presets=newton_sdf

**Multi-GPU (distributed) training** — for example on a cluster / OSMO workflow (substitute any task id above):

.. code-block:: bash

    ./isaaclab.sh -p -m torch.distributed.run --nnodes=1 --nproc_per_node=<NUM_GPUS> \
        scripts/reinforcement_learning/train.py --rl_library rsl_rl \
        --task <TASK_ID> \
        --num_envs <NUM_ENVS> \
        --distributed \
        agent.max_iterations=<MAX_ITERS> \
        --video --video_length 200 --video_interval 25600

The commands above train on the ``-ROS-Inference-v0`` ids so the trained environment already carries the
deployment contract. Swap in the plain ``...-NoJointVel-v0`` / ``...-TaskSpace-v0`` /
``...-TaskSpace-Newton-v0`` ids only for experiments you do not intend to deploy.

**Command breakdown:**

- ``--rl_library rsl_rl``: Selects the RSL-RL backend (required by the unified trainer)
- ``--num_envs 256``: Runs the default 256 parallel environments on each process / rank for the Newton profile
- Omitting ``--visualizer``: Uses headless execution by default for throughput
- ``--video_length 200`` / ``222``: Approximately one full episode for PhysX / Newton respectively
- ``--video_interval 76800``: Records every 150 iterations with 512 steps per environment
- ``--distributed``: Required when launching under ``torch.distributed.run``

Training uses a recurrent PPO agent (LSTM, 512 steps per environment). Existing PhysX profiles default to 1,500
iterations; the Newton reference defaults to 1,000. Videos are saved under ``logs/``.

.. note::

    **GPU and contact-capacity considerations**: PhysX keeps the base 4,096-environment default; the Newton task
    defaults to 256 environments per rank. Newton collision candidate capacity is scene-wide. If you increase the
    Newton environment count, also size ``max_triangle_pairs`` from observed warnings; overflow can omit candidate
    contacts. Reduce ``num_envs`` for out-of-memory or overflow failures.

**Monitoring Training Progress with TensorBoard:**

For PhysX profiles:

.. code-block:: bash

    ./isaaclab.sh -p -m tensorboard.main --logdir logs/rsl_rl/displayport_insertion_rizon4s

For the Newton profile:

.. code-block:: bash

    ./isaaclab.sh -p -m tensorboard.main --logdir logs/rsl_rl/displayport_insertion_rizon4s_newton_osc

Monitor ``Metrics/success_rate`` and reward curves to confirm learning. The curriculum anneals over the first 500 iterations — expect success rate to rise as the at-goal reset probability decreases.

.. _choosing-control-space:

Choosing a Control Space
~~~~~~~~~~~~~~~~~~~~~~~~~

The examples use the same recurrent PPO implementation and related insertion objectives. Individual profiles can
differ in observations, controller settings, reset distributions, and curriculum parameters. The PhysX profiles
run at 30 Hz; the Newton profile runs at approximately 33.3 Hz.

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * -
     - Joint space
     - Task space (recommended)
   * - Policy output
     - 7 joint-position deltas
     - 6-DoF Cartesian pose delta (OSC)
   * - Actor observation
     - 14 dims (or 21 with ``joint_vel``)
     - 18 dims (PhysX: TCP-first; Newton: socket-first with flange-origin pose)
   * - Real-robot requirement
     - Joint servo behavior must match sim; benefits from joint-level system identification
     - Task-space bridge that reproduces the selected OSC contract: TCP-observe / flange-control for PhysX, or
       flange-observe / flange-control for Newton
   * - Training ids
     - ``...-Grav-NoJointVel-ROS-Inference-v0`` (deployable), ``...-Grav-NoJointVel-v0``
     - ``...-Grav-TaskSpace-ROS-Inference-v0`` (PhysX) or
       ``...-Grav-TaskSpace-Newton-ROS-Inference-v0`` (Newton)

**Task space is the recommended route.** In our sim-to-real testing on the Flexiv Rizon 4s, task-space policies
transfer better than joint-space ones: commanding a Cartesian pose delta reduces sensitivity to differences in the
arm's joint servo behavior. It does not remove sensitivity to kinematic calibration, dynamics, controller tuning,
or frame conventions. Prefer it unless you specifically need joint-level control.

Use **joint space** when you must control the arm through its joint interface. Because the policy then commands
joints directly, transfer depends on the simulated robot matching yours closely — see the note below on robot
kinematics.

.. important::

   **Use a robot USD calibrated for the arm you will deploy.** Joint-space policies command joint positions, so any
   mismatch between the simulated kinematic parameters and your physical robot shows up directly as end-effector
   error during insertion — where the clearances are sub-millimetre. Task-space policies are less exposed to the
   joint-command mismatch, but still depend on accurate kinematics and frame transforms.

   Before training a joint-space policy, confirm the robot USD reflects your arm's measured kinematics. Flexiv
   publishes a per-robot calibration workflow for exporting an accurate robot description:
   `flexiv_calibration <https://github.com/flexivrobotics/flexiv_ros2/tree/release/lyrical-v1.9.3/flexiv_calibration>`__.
   Follow it to generate the calibrated description for your setup and convert it to USD. Derive the relevant
   environment configuration and replace ``scene.robot.spawn.usd_path``. The bundled Newton default is calibrated
   for the reference Rizon 4s with serial number ``063459``; it should not be assumed to match another arm. Override
   it from the command line with
   ``env.scene.robot.spawn.usd_path=/absolute/path/to/your_calibrated_robot.usd``.

In each case, train on the ``-ROS-Inference-v0`` id so the trained environment carries the deployment contract,
and use the matching ``-Play-v0`` id for evaluation and visualization. PhysX task space uses
``...-Grav-TaskSpace-ROS-Inference-v0`` / ``...-Grav-TaskSpace-Play-v0``; Newton uses
``...-Grav-TaskSpace-Newton-ROS-Inference-v0`` / ``...-Grav-TaskSpace-Newton-Play-v0``. Both task-space environments
use the LEAPP-exportable :class:`~isaaclab_tasks.contrib.deploy.mdp.DeployOperationalSpaceControllerActionCfg`
action, so a trained checkpoint can be exported directly (see :ref:`export-taskspace-leapp`).

Step 3: Export and Deploy on Real Robot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Recommended workflow:** export the trained policy with **LEAPP**, validate the export in simulation, then deploy the LEAPP package with Isaac ROS / Isaac Manipulator on the Flexiv robot.

Export from the ``-ROS-Inference-v0`` task id so the traced observation and action layout matches deployment. If
you followed the training steps above this is the same id you trained on, and no configuration swap is needed. In
joint space that is a **NoJointVel** variant (14-dim actor input). Task space uses
``-TaskSpace-ROS-Inference-v0`` for PhysX or ``-TaskSpace-Newton-ROS-Inference-v0`` for Newton; both have an 18-dim
actor input but different observation semantics.

Export with LEAPP (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`LEAPP <https://github.com/nvidia-isaac/leapp>`__ (Lightweight Export Annotations for Policy Pipelines) is the **default and recommended** path from a trained checkpoint to real-robot inference. It packages the policy together with input/output semantics (observation ordering, action scaling, recurrent LSTM state) so Isaac ROS deployment does not need to reimplement Isaac Lab preprocessing by hand.

**Prerequisites:** ``leapp>=0.5.2`` and a trained checkpoint for the matching ROS-inference task.

.. code-block:: bash

    ./isaaclab.sh -p -m pip install leapp

**Export the policy:**

.. tab-set::

   .. tab-item:: Joint space

      Uses the generic RSL-RL exporter:

      .. code-block:: bash

          ./isaaclab.sh -p scripts/reinforcement_learning/leapp/rsl_rl/export.py \
              --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-ROS-Inference-v0 \
              --checkpoint logs/rsl_rl/displayport_insertion_rizon4s/<run_timestamp>/model_<iteration>.pt

   .. tab-item:: Task space — PhysX

      Uses the DisplayPort exporter with ``--task_space_contract`` (see :ref:`export-taskspace-leapp`):

      .. code-block:: bash

          ./isaaclab.sh -p scripts/reinforcement_learning/leapp/rsl_rl/export_displayport_insertion.py \
              --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-ROS-Inference-v0 \
              --checkpoint logs/rsl_rl/displayport_insertion_rizon4s/<run_timestamp>/model_<iteration>.pt \
              --task_space_contract

   .. tab-item:: Task space — Newton

      Uses the same exporter with the Newton observation contract:

      .. code-block:: bash

          ./isaaclab.sh -p scripts/reinforcement_learning/leapp/rsl_rl/export_displayport_insertion.py \
              --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Newton-ROS-Inference-v0 \
              --checkpoint logs/rsl_rl/displayport_insertion_rizon4s_newton_osc/<run_timestamp>/model_<iteration>.pt \
              --task_space_contract \
              presets=newton_sdf

Replace ``<run_timestamp>`` and ``<iteration>`` with your training log path.

By default, export artifacts are written next to the checkpoint:

- Exported model (``.onnx`` by default, or ``.pt`` depending on backend)
- LEAPP metadata YAML describing the policy I/O graph
- Initial recurrent hidden state (``.safetensors``) — this policy uses an LSTM actor
- Pipeline graph visualization (``.png``)

Useful export flags:

- ``--export_method onnx-dynamo`` — default ONNX export backend
- ``--validation_steps 5`` — replay traced rollout data to verify the export (recommended; set ``0`` only for debugging)
- ``--export_save_path <dir>`` — write artifacts to a custom directory

See :doc:`Exporting Policies with LEAPP </source/policy_deployment/05_leapp/exporting_policies_with_leapp>` for full CLI options, backend choices, and troubleshooting.

**Validate before real-robot deployment.** Local validation currently differs by control space.

For a **joint-space LEAPP package**, ``play_displayport_insertion.py --leapp_model`` supports structured inference,
DP pose overrides, and ``policy_io.csv`` logging:

.. code-block:: bash

    ./isaaclab.sh -p scripts/reinforcement_learning/deploy/play_displayport_insertion.py \
        --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-ROS-Inference-v0 \
        --leapp_model logs/rsl_rl/displayport_insertion_rizon4s/<run_timestamp> \
        --num_envs 1 \
        --socket_pos 0.476 0.127 0.07 \
        --max_steps 200 \
        --log_dir logs/dp_inference_runs \
        --visualizer kit

``--leapp_model`` accepts either the LEAPP deploy YAML or the export directory that
contains it. Pose overrides and CSV logging still apply. Do not combine
``--leapp_model`` with ``--checkpoint`` or ``--replay_csv``.

For a **PhysX or Newton task-space package**, keep ``--validation_steps 5`` (or greater than zero) on the export
command so the exporter validates the traced graph and recurrent state. Then evaluate the original ``.pt``
checkpoint in simulation with its matching raw-checkpoint task id, as shown in
:ref:`displayport-deterministic-debugging`. The current ``play_displayport_insertion.py --leapp_model`` adapter
assumes joint observations
and absolute joint-target outputs, so it does not locally execute the six-dimensional processed pose delta emitted
by a task-space LEAPP package. Do not use it, or the generic deploy command below, as local task-space playback.

Generic joint-space LEAPP deploy (no DP pose / logging knobs):

.. code-block:: bash

    ./isaaclab.sh -p scripts/reinforcement_learning/leapp/deploy.py \
        --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-ROS-Inference-v0 \
        --leapp_model logs/rsl_rl/displayport_insertion_rizon4s/<run_timestamp>/<exported_leapp_yaml> \
        --viz kit

**Deploy on hardware:** pass the LEAPP export directory and metadata to your Isaac ROS / Isaac Manipulator workflow. Refer to the `Isaac ROS manipulation DNN policy documentation <https://nvidia-isaac-ros.github.io/reference_workflows/isaac_for_manipulation/packages/isaac_ros_manipulation_dnn_policy/index.html>`_ for on-robot setup. The on-robot pipeline typically includes:

1. **Perception** — socket pose estimation
2. **Motion planning** — approach trajectory to the insertion station (if used)
3. **Policy inference** — LEAPP-exported policy at control frequency in the ROS inference node
4. **Robot control** — Flexiv commands derived from policy actions: joint-position deltas in joint space, or a
   Cartesian pose delta applied through the task-space bridge in task space

The ROS inference environments define the deployment metadata LEAPP traces during export:

.. tab-set::

   .. tab-item:: Joint space

      ``Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-ROS-Inference-v0``

      - ``obs_order``: ``["arm_dof_pos", "socket_pos", "socket_quat"]``
      - ``policy_action_space``: ``"joint"``
      - ``observation_space``: 14
      - ``action_space``: 7
      - ``joint_action_scale``: 0.025

      Fixed deployment poses are set in ``config/displayport_rizon_4s/ros_inference_env_cfg.py``.

   .. tab-item:: Task space — PhysX

      ``Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-ROS-Inference-v0``

      - ``obs_order``: ``["eef_pos", "eef_rot_6d", "socket_kp_pos", "socket_kp_rot_6d"]``
      - ``policy_action_space``: ``"task"`` (``pose_rel``)
      - ``observation_space``: 18
      - ``action_space``: 6
      - ``action_scale``: 0.025 (metres for translation, radians for rotation)

      Fixed deployment poses are set in ``config/displayport_rizon_4s/task_space_ros_inference_env_cfg.py``.

   .. tab-item:: Task space — Newton

      ``Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Newton-ROS-Inference-v0``

      - ``obs_order``: ``["socket_pos", "tool_pos", "tool_rot_6d", "socket_rot_6d"]``
      - ``policy_action_space``: ``"task"`` (``pose_rel``)
      - ``observation_space``: 18
      - ``action_space``: 6
      - ``action_scale``: 0.025 (metres for translation, radians for rotation)

      Fixed deployment poses are set in
      ``config/displayport_rizon_4s/task_space_newton_ros_inference_env_cfg.py``.

.. _export-taskspace-leapp:

Exporting a Task-Space Policy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **task-space** checkpoint is exported against its matching ROS-inference task: ``...-Grav-TaskSpace-ROS-Inference-v0``
for PhysX or ``...-Grav-TaskSpace-Newton-ROS-Inference-v0`` for Newton. Use the DisplayPort-specific exporter
:file:`scripts/reinforcement_learning/leapp/rsl_rl/export_displayport_insertion.py` and its
``--task_space_contract`` flag, because the action is a scaled 6-DoF Cartesian pose delta rather than joint targets.
The script reuses the generic RSL-RL export flow and only substitutes the export routine:

.. tab-set::

   .. tab-item:: PhysX

      .. code-block:: bash

          ./isaaclab.sh -p scripts/reinforcement_learning/leapp/rsl_rl/export_displayport_insertion.py \
              --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-ROS-Inference-v0 \
              --checkpoint logs/rsl_rl/displayport_insertion_rizon4s/<run_timestamp>/model_<iteration>.pt \
              --export_save_path <output_dir> \
              --task_space_contract

   .. tab-item:: Newton

      .. code-block:: bash

          ./isaaclab.sh -p scripts/reinforcement_learning/leapp/rsl_rl/export_displayport_insertion.py \
              --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Newton-ROS-Inference-v0 \
              --checkpoint logs/rsl_rl/displayport_insertion_rizon4s_newton_osc/<run_timestamp>/model_<iteration>.pt \
              --export_save_path <output_dir> \
              --task_space_contract \
              presets=newton_sdf

The export writes a ``DisplayPortTaskSpace`` LEAPP package (``.onnx`` graph, ``.yaml`` semantics, and
``_initial_values.safetensors`` for the recurrent state). The exporter derives named inputs from the selected
task's observation metadata and rejects invalid 18-D contracts before tracing:

- **Deploy inputs:** ``eef_pos`` (3), ``eef_rot_6d`` (6), ``socket_kp_pos`` (3), and
  ``socket_kp_rot_6d`` (6), plus LSTM state. Both backends expose these canonical port names and sources.
- **Newton actor order:** ``socket_kp_pos`` (3), ``eef_pos`` (3), ``eef_rot_6d`` (6), then
  ``socket_kp_rot_6d`` (6). The exporter maps the training terms ``socket_pos``, ``tool_pos``,
  ``tool_rot_6d``, and ``socket_rot_6d`` to the canonical Deploy ports without changing their slices or
  reordering the vector presented to the actor.
- **Output:** ``arm_action`` (6) — a ``pose_rel`` Cartesian delta ``[dx, dy, dz, dθx, dθy, dθz]`` consumed by the
  task-space bridge (``isaaclab_connection: action:arm_action:pose_rel``). The exporter accepts either one scalar or
  three per-axis values for each OSC position and orientation scale and bakes those configured scales into this output.

The Newton policy observes and controls the Flexiv ``flange`` origin with a zero body offset. Configure the hardware
bridge's EEF pose source and relative-pose action frame to that same origin; do not add a legacy TCP translation (such
as 0.15 m) unless the observation and action configuration used during training is changed to match it. The socket
input remains the insertion keypoint, offset 0.0375 m along the socket's local positive x-axis.

For either backend, keep exporter ``--validation_steps`` greater than zero and separately evaluate the original
``.pt`` checkpoint in the matching simulator. The current local ``--leapp_model`` adapters are joint-space only;
they do not validate or play the processed six-dimensional task-space output. Pass the exported package to the
matching Isaac ROS task-space bridge for hardware deployment.

Alternative: Raw Checkpoint Deployment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For development or legacy Isaac Manipulator setups, you can deploy the RSL-RL checkpoint directly without a LEAPP export step. This path uses the ``.pt`` checkpoint with ``agent.yaml`` and ``env.yaml`` from:

.. code-block:: text

    PhysX:  logs/rsl_rl/displayport_insertion_rizon4s/<run_timestamp>/model_<iteration>.pt
    Newton: logs/rsl_rl/displayport_insertion_rizon4s_newton_osc/<run_timestamp>/model_<iteration>.pt

This is **not recommended for shipping** — you must manually ensure observation ordering, action scaling, and LSTM state handling match training. Prefer the LEAPP export path above for production deployment.


Troubleshooting
---------------

PhysX Collision Stack Overflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Error Message:**

.. code-block:: text

    PhysX error: PxGpuDynamicsMemoryConfig::collisionStackSize buffer overflow detected

**Cause:** GPU collision buffer is too small for contact-rich plug/socket interaction across many parallel environments.

**Solution:** Increase ``gpu_collision_stack_size`` in ``displayport_insertion_env_cfg.py`` (default is ``2**30``):

.. code-block:: python

    sim: SimulationCfg = SimulationCfg(
        physics=PhysxCfg(
            gpu_collision_stack_size=2**31,  # Increase if overflow persists
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )

CUDA Out of Memory
~~~~~~~~~~~~~~~~~~

**Solutions (in order of preference):**

1. Reduce parallel environments:

   .. code-block:: bash

       ./isaaclab.sh train --rl_library rsl_rl \
           --task <TASK_ID> \
           --num_envs 128

2. Reduce plug/socket ``solver_position_iteration_count`` in ``displayport_insertion_env_cfg.py`` (trade-off: more penetration)

3. Disable video recording during training


.. _displayport-deterministic-debugging:

Deterministic Debugging (Play / Inference Script)
-------------------------------------------------

For raw RSL-RL checkpoints, prefer the dedicated play script over the generic ``./isaaclab.sh play`` path. It keeps
DP-specific pose overrides, perception-error injection, and ``policy_io.csv`` logging out of the shared entrypoint.

Raw checkpoint playback supports joint space and both task-space backends. ``--leapp_model`` playback in this
script is limited to joint-space packages, as described in Step 3.

**Raw RSL-RL checkpoint:**

.. tab-set::

   .. tab-item:: Joint space

      .. code-block:: bash

          ./isaaclab.sh -p scripts/reinforcement_learning/deploy/play_displayport_insertion.py \
              --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-ROS-Inference-v0 \
              --checkpoint logs/rsl_rl/displayport_insertion_rizon4s/<run>/model_<iteration>.pt \
              --num_envs 1 \
              --socket_pos 0.476 0.127 0.07 \
              --observed_socket_pos 0.486 0.127 0.07 \
              --max_steps 200 \
              --log_dir logs/dp_inference_runs \
              --visualizer kit

   .. tab-item:: Task space — PhysX

      .. code-block:: bash

          ./isaaclab.sh -p scripts/reinforcement_learning/deploy/play_displayport_insertion.py \
              --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Play-v0 \
              --checkpoint logs/rsl_rl/displayport_insertion_rizon4s/<run>/model_<iteration>.pt \
              --num_envs 1 \
              --socket_pos 0.476 0.127 0.07 \
              --max_steps 200 \
              --log_dir logs/dp_inference_runs \
              --visualizer kit

   .. tab-item:: Task space — Newton

      .. code-block:: bash

          ./isaaclab.sh -p scripts/reinforcement_learning/deploy/play_displayport_insertion.py \
              --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Newton-Play-v0 \
              --checkpoint logs/rsl_rl/displayport_insertion_rizon4s_newton_osc/<run>/model_<iteration>.pt \
              --num_envs 1 \
              --socket_pos 0.476 0.127 0.07 \
              --max_steps 200 \
              --log_dir logs/dp_inference_runs \
              --visualizer kit \
              presets=newton_sdf

**Exact first joint-space observation from real/sim CSV** (closed-loop; injects ``obs_*`` from the chosen row as
the first policy input, then continues with live sim observations):

.. code-block:: bash

    ./isaaclab.sh -p scripts/reinforcement_learning/deploy/play_displayport_insertion.py \
        --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-ROS-Inference-v0 \
        --checkpoint logs/rsl_rl/displayport_insertion_rizon4s/<run>/model_<iteration>.pt \
        --init_obs_csv rollouts/sim_real_compare/real_policy_gt_sock.csv \
        --socket_pos 0.475 0.125 0.07 \
        --num_envs 1 \
        --max_steps 200 \
        --log_dir logs/dp_init_obs_runs \
        --visualizer kit

Unset ``--robot_joint_pos`` / ``--observed_socket_*`` are seeded from that CSV row so the
scene starts aligned. Use ``--init_obs_step N`` to pick a non-zero row. This joint-space path also works with a
joint-space ``--leapp_model``; it is not compatible with ``--replay_csv``.

**Joint-space LEAPP-exported ONNX policy** uses the same script; see Step 3 for ``--leapp_model``.

**Joint-space open-loop replay** of a logged ``policy_io.csv`` / real rollout:

.. code-block:: bash

    ./isaaclab.sh -p scripts/reinforcement_learning/deploy/play_displayport_insertion.py \
        --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-ROS-Inference-v0 \
        --replay_csv rollouts/sim_real_compare/real_policy.csv \
        --socket_pos 0.475 0.125 0.07 \
        --log_dir rollouts/sim_real_compare/replay_run \
        --visualizer kit

Pose conventions:

- ``--socket_pos`` / ``--observed_socket_pos`` are socket **insertion-geometry**
  world positions (mate point), matching ``fixed_asset_init_pos_center``, not the
  USD root. The USD root is derived via ``compute_socket_root``.
- Quaternions are ``(x, y, z, w)``.
- If ``--observed_socket_*`` is omitted, the policy observes the true simulated socket pose.
- Telemetry is written to ``policy_io.csv`` (+ ``run_config.json``) under ``--log_dir``
  or ``<checkpoint_dir>/inference_logs/<timestamp>/``.

Generic play (no DP pose / CSV knobs) still works for a quick smoke test:

.. tab-set::

   .. tab-item:: Joint space

      .. code-block:: bash

          ./isaaclab.sh play --rl_library rsl_rl \
              --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-Play-v0 \
              --num_envs 1 \
              --checkpoint <path_to_model.pt>

   .. tab-item:: Task space — PhysX

      .. code-block:: bash

          ./isaaclab.sh play --rl_library rsl_rl \
              --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Play-v0 \
              --num_envs 1 \
              --checkpoint <path_to_model.pt>

   .. tab-item:: Task space — Newton

      .. code-block:: bash

          ./isaaclab.sh play --rl_library rsl_rl \
              --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Newton-Play-v0 \
              --num_envs 1 \
              --checkpoint <path_to_model.pt> \
              presets=newton_sdf

To match a specific real-world station layout, edit the workspace constants in
``config/displayport_rizon_4s/joint_pos_env_cfg.py`` (training layout) or
``config/displayport_rizon_4s/ros_inference_env_cfg.py`` (deployment layout). Task-space variants use the
corresponding ``task_space*_env_cfg.py`` and ``task_space*_ros_inference_env_cfg.py`` files.

.. code-block:: python

    # Training station layout (joint_pos_env_cfg.py)
    _GEOMETRY_POS = (0.475, 0.125, 0.06)
    _SOCKET_ROT = (0.5, 0.5, 0.5, -0.5)

    # Deployment layout (ros_inference_env_cfg.py)
    _DEPLOY_GEOMETRY_POS = (0.476, 0.127, 0.07)
    _DEPLOY_SOCKET_ROT = (0.5, 0.5, 0.5, -0.5)

This workflow is useful for:

- Comparing simulated and real-world policy behavior at a known socket pose
- Validating joint-space LEAPP playback and task-space exporter compilation before hardware deployment
- Verifying plug grasp and approach trajectories before full perception integration
- Debugging insertion failures at a fixed station configuration


Further Resources
-----------------

- Gear Assembly Sim-to-Real Tutorial: :ref:`walkthrough_sim_to_real`
- Exporting Policies with LEAPP: :doc:`/source/policy_deployment/05_leapp/exporting_policies_with_leapp`
- `Isaac ROS Manipulation Documentation <https://nvidia-isaac-ros.github.io/reference_workflows/isaac_for_manipulation/index.html>`_
- RL Training Tutorial: :ref:`tutorial-run-rl-training`
