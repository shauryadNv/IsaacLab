Added
^^^^^

* Added an optional ``load_training_state`` hook to the RSL-RL training entry point. After a checkpoint is loaded on
  resume, the entry point calls ``env.unwrapped.load_training_state(resume_path)`` when the environment defines it, so
  an environment can restore training state that the RSL-RL checkpoint does not carry, such as a domain-randomization
  curriculum level. Environments without the hook are unaffected.

Fixed
^^^^^

* Fixed resumed RSL-RL runs taking their first updates at the configured initial learning rate. The optimizer state,
  including its decayed learning rate, is restored from the checkpoint, but PPO's own ``learning_rate`` (which the
  adaptive schedule writes back into the optimizer) restarted at the config value, so a run resumed after its rate had
  decayed collapsed on its first update. The entry point now resumes from the optimizer's restored learning rate.
