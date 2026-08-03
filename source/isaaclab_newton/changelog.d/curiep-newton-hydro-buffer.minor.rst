Added
^^^^^

* Added ``HydroelasticSDFCfg.buffer_mult_broad`` and
  ``HydroelasticSDFCfg.buffer_mult_iso`` to size Newton's hydroelastic
  broad-phase and iso-surface buffers independently.

Fixed
^^^^^

* Fixed Newton inverse-kinematics actions invalidating live collision resources
  when constructing their prototype model.
