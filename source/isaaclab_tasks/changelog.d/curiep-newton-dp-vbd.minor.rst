Added
^^^^^

* Added all-VBD and mixed MJWarp/VBD proxy and ADMM physics presets for the
  DisplayPort insertion task.

Fixed
^^^^^

* Fixed scalar command-line overrides for preset-valued task fields.
* Fixed all-VBD DisplayPort parity by compensating gravity on robot bodies
  while retaining world gravity for manipulated objects.
* Fixed the DisplayPort VBD timing presets to use the validated 2 kHz solver,
  200 Hz collision, and 33.3 Hz policy cadence.
* Increased the DisplayPort ADMM contact capacity to cover measured randomized
  rollout demand without discarding collision candidates.
