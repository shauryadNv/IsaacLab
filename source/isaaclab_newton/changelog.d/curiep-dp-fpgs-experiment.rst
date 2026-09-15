Added
^^^^^

* Added :class:`~isaaclab_newton.physics.FeatherPGSSolverCfg` and its Newton
  physics manager for experimental FeatherPGS simulation.
* Added opt-in FeatherPGS constraint-row watermark reporting during debug runs
  and at shutdown.

Fixed
^^^^^

* Preserved per-environment entity labels when cloning with Newton builds that
  predate batched ``label_prefixes`` support.
