# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for deployment-specific observation noise models."""

import torch

from isaaclab.utils.noise import UniformNoiseCfg
from isaaclab.utils.noise.noise_model import uniform_noise

from isaaclab_tasks.contrib.deploy.mdp.noise_models import (
    ResetSampledConstantNoiseModel,
    ResetSampledConstantNoiseModelCfg,
)


def _make_noise_model(*, sample_per_component: bool | None = None) -> ResetSampledConstantNoiseModel:
    noise_cfg = UniformNoiseCfg(func=uniform_noise, n_min=-1.0, n_max=1.0)
    if sample_per_component is None:
        cfg = ResetSampledConstantNoiseModelCfg(noise_cfg=noise_cfg)
    else:
        cfg = ResetSampledConstantNoiseModelCfg(
            noise_cfg=noise_cfg,
            sample_per_component=sample_per_component,
        )
    return ResetSampledConstantNoiseModel(cfg, num_envs=4, device="cpu")


def test_reset_sampled_constant_noise_preserves_scalar_broadcast_by_default():
    """Legacy mode should draw one held scalar per environment and broadcast it."""
    model = _make_noise_model()
    data = torch.zeros((4, 3))

    assert model._noise_model_cfg.sample_per_component is False

    torch.manual_seed(123)
    expected_scalar = torch.rand((4, 1)) * 2.0 - 1.0
    torch.manual_seed(123)
    model.reset()

    sampled = model(data)
    torch.testing.assert_close(sampled, expected_scalar.repeat(1, 3))
    torch.testing.assert_close(model(data), sampled)

    selected_envs = [1, 3]
    torch.manual_seed(456)
    expected_selected = torch.rand((2, 1)) * 2.0 - 1.0
    torch.manual_seed(456)
    model.reset(selected_envs)

    expected = sampled.clone()
    expected[selected_envs] = expected_selected.repeat(1, 3)
    torch.testing.assert_close(model(data), expected)


def test_reset_sampled_constant_noise_can_sample_and_hold_independent_components():
    """Opt-in mode should sample every component once and selectively reset rows."""
    model = _make_noise_model(sample_per_component=True)
    data = torch.zeros((4, 3))

    torch.manual_seed(123)
    model.reset()
    sampled = model(data)
    torch.manual_seed(123)
    expected_initial = torch.rand((4, 3)) * 2.0 - 1.0

    torch.testing.assert_close(sampled, expected_initial)
    assert torch.any(sampled[:, 1:] != sampled[:, :-1])
    torch.testing.assert_close(model(data), sampled)

    selected_envs = [1, 3]
    torch.manual_seed(456)
    expected_selected = torch.rand((2, 3)) * 2.0 - 1.0
    torch.manual_seed(456)
    model.reset(selected_envs)

    expected = sampled.clone()
    expected[selected_envs] = expected_selected
    torch.testing.assert_close(model(data), expected)
