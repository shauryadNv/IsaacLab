# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton collision-pipeline configuration."""

from isaaclab_newton.physics import HydroelasticSDFCfg, NewtonCollisionPipelineCfg


def test_hydroelastic_buffer_multipliers_reach_newton():
    """The per-stage buffer multipliers should be forwarded to Newton unchanged."""
    cfg = NewtonCollisionPipelineCfg(
        sdf_hydroelastic_config=HydroelasticSDFCfg(
            buffer_fraction=0.125,
            buffer_mult_broad=8,
            buffer_mult_iso=2,
        )
    )

    hydro_cfg = cfg.to_pipeline_args()["sdf_hydroelastic_config"]

    assert hydro_cfg.buffer_fraction == 0.125
    assert hydro_cfg.buffer_mult_broad == 8
    assert hydro_cfg.buffer_mult_iso == 2


def test_contact_matching_reaches_newton():
    """The collision-pipeline matching mode should be forwarded unchanged."""
    cfg = NewtonCollisionPipelineCfg(contact_matching="latest")

    assert cfg.to_pipeline_args()["contact_matching"] == "latest"
