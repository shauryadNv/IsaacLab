# Corrected no-near-goal ADR follow-up

This sweep uses payload gravity compensation, capped OSC-gain randomization, no
near-goal resets, checkpoint-paired ADR state, and the validated PhysX joint
Coulomb-friction range.

Submit the seed-42 controls first:

```bash
./submit.sh all_off_s42 initial
./submit.sh all_on_s42 initial
./submit.sh all_on_sdstd_s42 initial
./submit.sh joint_friction_s42 initial
```

`all_on_s42` and `all_on_sdstd_s42` differ only in the policy standard
deviation parameterization. The isolated joint-friction run validates the
corrected dry-friction semantics. Submit `all_on_s123`, `all_on_s321`, and
`plug_mass_s123` only after the seed-42 workflows show healthy rollout,
curriculum, and checkpoint telemetry.

The submission script refuses an existing `name + attempt` pair in
`runs.tsv`. Use a higher `ATTEMPT` only for an intentional resume.
