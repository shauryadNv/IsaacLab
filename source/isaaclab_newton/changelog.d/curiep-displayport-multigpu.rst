Fixed
^^^^^

* Fixed CUDA graph capture and replay selecting the default GPU instead of
  the physics manager device during distributed Newton training.

* Fixed distributed hydroelastic training lazily loading collision
  kernels during the first simulation step.
