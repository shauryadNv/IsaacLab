Added
^^^^^

* Added optional ``save_training_state`` and ``load_training_state`` hooks to the RSL-RL training entry point. The
  entry point pairs environment-owned state with every checkpoint and restores it after loading, allowing an
  environment to preserve state that RSL-RL does not carry, such as a domain-randomization curriculum level.
  Environments without the hooks are unaffected.

Fixed
^^^^^

* Fixed resumed RSL-RL runs taking their first updates at the configured initial learning rate. The optimizer state,
  including its decayed learning rate, is restored from the checkpoint, but PPO's own ``learning_rate`` (which the
  adaptive schedule writes back into the optimizer) restarted at the config value, so a run resumed after its rate had
  decayed collapsed on its first update. The entry point now resumes from the optimizer's restored learning rate.
* Fixed resumed RSL-RL runs repeating the checkpoint's already completed learning iteration. Training now continues
  at the following iteration, keeping checkpoint numbering and rollout schedules monotonic across resumes.
