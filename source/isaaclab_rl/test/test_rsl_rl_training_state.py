# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for environment-owned state paired with RSL-RL checkpoints."""

from types import SimpleNamespace

from isaaclab_rl.entrypoints.backends.train_rsl_rl import (
    _advance_runner_after_checkpoint_load,
    _enable_training_state_checkpointing,
)


def test_runner_checkpoint_saves_environment_training_state_after_model():
    """Every runner checkpoint is followed by its matching environment state save."""
    calls = []
    runner = SimpleNamespace(save=lambda path, infos=None: calls.append(("model", path, infos)))
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            save_training_state=lambda path: calls.append(("environment", path)),
        )
    )

    _enable_training_state_checkpointing(runner, env)
    runner.save("model_100.pt", infos={"value": 1})

    assert calls == [
        ("model", "model_100.pt", {"value": 1}),
        ("environment", "model_100.pt"),
    ]


def test_resumed_runner_starts_after_the_completed_checkpoint_iteration():
    """A checkpoint from iteration N must resume at N+1 rather than replaying N."""
    runner = SimpleNamespace(current_learning_iteration=100)

    _advance_runner_after_checkpoint_load(runner)

    assert runner.current_learning_iteration == 101
