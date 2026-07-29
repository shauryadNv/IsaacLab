# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton collision-pipeline configuration."""

from isaaclab_newton.physics import HydroelasticSDFCfg, NewtonCollisionPipelineCfg


def test_hydroelastic_broad_buffer_multiplier_reaches_newton():
    """The broad-phase multiplier should be forwarded to Newton unchanged."""
    cfg = NewtonCollisionPipelineCfg(
        sdf_hydroelastic_config=HydroelasticSDFCfg(
            buffer_fraction=0.125,
            buffer_mult_broad=8,
        )
    )

    hydro_cfg = cfg.to_pipeline_args()["sdf_hydroelastic_config"]

    assert hydro_cfg.buffer_fraction == 0.125
    assert hydro_cfg.buffer_mult_broad == 8
