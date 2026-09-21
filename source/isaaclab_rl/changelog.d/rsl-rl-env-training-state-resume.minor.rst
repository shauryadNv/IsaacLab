Added
^^^^^

* Added an optional ``load_training_state`` hook to the RSL-RL training entry point. After a checkpoint is loaded on
  resume, the entry point calls ``env.unwrapped.load_training_state(resume_path)`` when the environment defines it, so
  an environment can restore training state that the RSL-RL checkpoint does not carry, such as a domain-randomization
  curriculum level. Environments without the hook are unaffected.
