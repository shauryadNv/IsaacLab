Experimental Geometric-Fabric Controller
========================================

The DisplayPort task includes an isolated, **experimental** geometric-fabric
control path for evaluating smoother policy-to-robot motion. It is not a
replacement for the supported joint-space or operational-space-controller
tasks, and it is not yet a deployment configuration.

Scope and controller contract
-----------------------------

``IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Fabric`` uses PhysX
with three nested rates: 240 Hz physics and explicit joint tracking, 60 Hz
artificial-state integration, and a 30 Hz policy. The policy emits a normalized
six-dimensional TCP wrench in the robot-base frame. The default force/torque
scales are ``(0.5, 0.5, 0.5) N`` and ``(0.05, 0.05, 0.05) N*m``.

The actor receives 46 values in this fixed order: TCP position (3), TCP 6D
rotation (6), socket position (3), socket 6D rotation (6), measured joint
position and velocity (7 each), and artificial fabric position and velocity
(7 each). Socket state is ground truth from simulation; there is no sensor
noise or latency model in this experiment. The critic also has 46 values, but
uses privileged plug state. Changing either order invalidates a checkpoint.

The clean-room controller was implemented from the equations described in the
FABRICS paper; no code from the separately licensed NVLabs implementation is
vendored. It combines an HD2 socket-transverse geometry, a root metric,
posture and joint-limit terms, damping, and a bounded policy-wrench pullback.
It deliberately differs from the paper implementation in several ways:

* hard limits use a componentwise feasible acceleration intersection whose
  quadratic-velocity and cubic-position Bernstein control points conservatively
  certify the complete continuous segment, not only its endpoints;
* the joint-limit term is a bounded smooth barrier;
* fabric integration and 240 Hz tracker interpolation use exact constant-jerk
  segments instead of the paper's approximate RK2 update, producing continuous
  position, velocity, and acceleration targets; every 240 Hz target sample is
  checked again, clamped for numerical roundoff, and held with a persistent
  fault on a material position or velocity limit violation;
* the tracker applies explicit PD plus ``M(q) * qdd_f + g(q)`` and omits
  Coriolis effort because it is not exposed by the current PhysX tensor API;
* insertion-axis motion remains policy controlled while the fabric geometry
  constrains only the two socket-transverse directions.

The task is PhysX-only. Physical gravity is enabled and compensated exactly
once by the tracker. Do not switch the backend, disable gravity compensation,
add an independently sampled socket observation, or change the 30/60/240 Hz
rate contract without revalidation.

Training gate
-------------

Run the tracked validator with the exact Isaac Sim 6.0.1 runtime before any
training submission:

.. code-block:: bash

   uv run --extra isaacsim python \
       scripts/reinforcement_learning/deploy/validate_displayport_fabric.py \
       --task IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-Fabric \
       --num_envs 4 \
       --visualizer none

The command exits nonzero on a runtime mismatch, observation-order or shape
error, dynamics or three-pose kinematics mismatch, fabric numerical fault,
tracking-health violation, an unstable plug at the default 4 mm shallow-engagement
contact pose, or incorrect scripted insertion/search response. It writes
``artifacts/displayport_fabric_validation.json`` by default.

Repeat the same command with ``--visualizer kit`` and inspect the zero-input,
pre-inserted, and scripted phases. A passing headless report does not replace
this visual check. Next run a short single-GPU training diagnostic and inspect
the logged ``Fabric/*`` telemetry before scaling the experiment. In particular,
require zero tracker position/velocity limit violations and inspect measured
physical joint acceleration and jerk. The actual-effort fields sample the most
recent actuator output at each tracker tick; they are distinct from the
predicted effort for the command being written on that tick.

Real robot status
-----------------

There is currently no matching real-time Flexiv controller in this repository.
The existing ROS joint-impedance path accepts position commands and is not the
same controller. Hardware deployment requires a dedicated real-time interface
that evaluates identical kinematics and fabric state, interpolates 60 Hz fabric
targets to the robot servo rate, applies the same joint limits and gravity
contract, and provides watchdog, effort, contact, and stale-command protection.
Do not deploy a checkpoint from this task on hardware until that controller and
its non-contact safety validation exist.

See the `FABRICS paper <https://arxiv.org/html/2405.02250v1>`_ for the research
method and the `NVLabs reference repository <https://github.com/NVlabs/FABRICS>`_
for the separately licensed reference implementation.
