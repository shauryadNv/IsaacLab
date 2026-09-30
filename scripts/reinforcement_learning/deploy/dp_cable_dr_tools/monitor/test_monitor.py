# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral tests for cumulative ADR checkpoint progress."""

import pytest
from monitor import (
    checkpoint_progress,
    checkpoint_sort_key,
    remaining_iterations,
    save_checkpoint_cache,
    sync_attempt,
)


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
        {"checkpoint": "model_10.pt", "step": 10 * 512},
        {"checkpoint": "model_10.pt", "step": -512},
        {"checkpoint": "model_10.pt", "step": 512 + 1},
    ],
)
def test_checkpoint_progress_rejects_mismatched_or_invalid_sidecars(state):
    """Only exact, positive rollout-boundary sidecars are authoritative."""
    with pytest.raises(ValueError):
        checkpoint_progress("model_10.pt", state, 512)


def test_sync_attempt_recovers_cached_progress_when_swift_listing_fails(tmp_path, monkeypatch):
    """A transient remote-listing failure must not erase a mirrored attempt's progress."""
    checkpoint = {
        **checkpoint_progress(
            "model_50.pt",
            {"checkpoint": "model_50.pt", "step": 252 * 512},
            512,
        ),
        "file": "model_50.pt",
        "folder": "run-folder",
        "url": "https://example/model_50.pt",
        "state_file": "model_50.adr_state.json",
        "state_url": "https://example/model_50.adr_state.json",
        "resume_from": "commit/tag/task/run-folder",
        "attempt": 2,
    }
    save_checkpoint_cache(tmp_path, checkpoint)
    monkeypatch.setattr("monitor.swift_list", lambda _url: [])

    recovered = sync_attempt(
        {
            "COMMIT": "commit",
            "STEPS_PER_ITERATION": "512",
            "SWIFT_HTTP": "https://example",
            "ROBOT_TYPE": "rizon4s",
        },
        {"attempt": "2", "run_tag": "/tag", "commit": "commit"},
        tmp_path,
        {},
    )

    assert recovered == checkpoint
