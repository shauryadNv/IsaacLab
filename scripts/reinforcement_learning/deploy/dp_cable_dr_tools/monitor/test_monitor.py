# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral tests for cumulative ADR checkpoint progress."""

import pytest

from monitor import checkpoint_progress, checkpoint_sort_key, remaining_iterations


def test_checkpoint_progress_uses_the_paired_cumulative_step():
    """A reset model suffix must not replace cumulative training progress."""
    progress = checkpoint_progress(
        "model_50.pt",
        {"checkpoint": "model_50.pt", "step": 252 * 512},
        512,
    )

    assert progress == {
        "iteration": 251,
        "model_iteration": 50,
        "completed_iterations": 252,
        "common_step_counter": 252 * 512,
        "iteration_offset": 201,
    }


def test_checkpoint_selection_prefers_cumulative_progress_then_attempt():
    """A later resumed checkpoint wins even when its raw model suffix is smaller."""
    older = {
        **checkpoint_progress("model_200.pt", {"checkpoint": "model_200.pt", "step": 201 * 512}, 512),
        "attempt": 1,
    }
    newer = {
        **checkpoint_progress("model_50.pt", {"checkpoint": "model_50.pt", "step": 252 * 512}, 512),
        "attempt": 2,
    }

    assert checkpoint_sort_key(newer) > checkpoint_sort_key(older)
    assert remaining_iterations(6000, newer) == 5748


@pytest.mark.parametrize(
    "state",
    [
        {"checkpoint": "model_9.pt", "step": 10 * 512},
        {"checkpoint": "model_10.pt", "step": -512},
        {"checkpoint": "model_10.pt", "step": 512 + 1},
    ],
)
def test_checkpoint_progress_rejects_mismatched_or_invalid_sidecars(state):
    """Only exact, positive rollout-boundary sidecars are authoritative."""
    with pytest.raises(ValueError):
        checkpoint_progress("model_10.pt", state, 512)
