Fixed
^^^^^

* Fixed :class:`~isaaclab.actuators.ActuatorBase` to broadcast USD-template actuator
  parameters supplied with shape ``[1, num_joints]`` (as the Newton backend reads from a
  single prototype instance) to all environments instead of raising ``ValueError``.
* Fixed :class:`~isaaclab.app.AppLauncher` Isaac Sim version detection to not raise
  ``NameError`` when ``isaacsim`` is not importable, as in omni-style installs.
* Fixed asset resolution to fall back to a direct HTTP download and to skip non-essential
  dependencies (materials, textures) whose download fails, so headless and kitless
  workflows can proceed.
