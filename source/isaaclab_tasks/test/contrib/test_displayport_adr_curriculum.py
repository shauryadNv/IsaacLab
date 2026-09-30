# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral tests for DisplayPort automatic domain randomization state."""

import json
from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.contrib.deploy.cable_insertion.insertion_env import DisplayportInsertionEnv
from isaaclab_tasks.contrib.deploy.mdp.dr_curriculum import SuccessDifficultyScheduler


def _make_scheduler(tmp_path, step: int = 0) -> SuccessDifficultyScheduler:
    scheduler = object.__new__(SuccessDifficultyScheduler)
    scheduler._env = SimpleNamespace(device=torch.device("cpu"), common_step_counter=step)
    scheduler.num_levels = 50
    scheduler.success_threshold = 0.4
    scheduler.min_episodes_between = 5.0
    scheduler._min_steps_between = None
    scheduler.demote = False
    scheduler.smoothing = 0.1
    scheduler.at_goal_event = None
    scheduler._spawned_at_goal = None
    scheduler._spawned_at_goal_resolved = True
    scheduler.level = 0
    scheduler._success_rate = 0.0
    scheduler._last_change_step = 0
    scheduler._skip_next_update = False
    scheduler._state_path = str(tmp_path / "adr_state.json")
    return scheduler


def test_checkpoint_sidecars_restore_the_state_for_the_requested_checkpoint(tmp_path):
    """An older checkpoint must not load the curriculum level reached by a newer one."""
    scheduler = _make_scheduler(tmp_path, step=100)
    old_checkpoint = tmp_path / "model_100.pt"
    scheduler.level = 7
    scheduler._success_rate = 0.45
    assert scheduler.save_for_checkpoint(str(old_checkpoint))

    scheduler._env.common_step_counter = 200
    scheduler.level = 23
    scheduler._success_rate = 0.91
    assert scheduler.save_for_checkpoint(str(tmp_path / "model_200.pt"))

    resumed = _make_scheduler(tmp_path / "resumed", step=0)
    assert resumed.restore_from_checkpoint(str(old_checkpoint))
    assert resumed.level == 7
    assert resumed._env.common_step_counter == 100
    assert resumed._success_rate == pytest.approx(0.45)
    assert resumed._last_change_step == 100
    assert resumed._skip_next_update

    state = json.loads((tmp_path / "model_100.adr_state.json").read_text())
    assert state["checkpoint"] == "model_100.pt"
    assert state["step"] == 100


def test_restore_rejects_a_latest_state_written_after_the_checkpoint(tmp_path):
    """A mutable latest-state file cannot move an older checkpoint into the future."""
    (tmp_path / "adr_state.json").write_text(
        json.dumps({"version": 2, "level": 30, "num_levels": 50, "success_rate": 0.9, "step": 200})
    )
    scheduler = _make_scheduler(tmp_path / "resumed", step=100)
    scheduler.level = 3

    assert not scheduler.restore_from_checkpoint(str(tmp_path / "model_100.pt"))
    assert scheduler.level == 3


def test_restore_rejects_a_stale_latest_state_for_a_newer_checkpoint(tmp_path):
    """A latest-state fallback must identify the exact checkpoint rollout boundary."""
    (tmp_path / "adr_state.json").write_text(
        json.dumps({"version": 2, "level": 8, "num_levels": 50, "success_rate": 0.5, "step": 50})
    )
    scheduler = _make_scheduler(tmp_path / "resumed", step=100)
    scheduler.level = 3

    assert not scheduler.restore_from_checkpoint(str(tmp_path / "model_99.pt"))
    assert scheduler.level == 3


def test_restore_accepts_latest_state_at_the_exact_checkpoint_step(tmp_path):
    """An exact-step latest-state fallback is safe when a paired sidecar is unavailable."""
    (tmp_path / "adr_state.json").write_text(
        json.dumps({"version": 2, "level": 8, "num_levels": 50, "success_rate": 0.5, "step": 100})
    )
    scheduler = _make_scheduler(tmp_path / "resumed", step=100)

    assert scheduler.restore_from_checkpoint(str(tmp_path / "model_99.pt"))
    assert scheduler.level == 8
    assert scheduler._success_rate == pytest.approx(0.5)


def test_restore_step_counter_uses_the_completed_rollout_boundary(tmp_path):
    """Checkpoint iteration N represents N+1 completed RSL-RL rollouts."""
    checkpoint = tmp_path / "model_100.pt"
    torch.save({"iter": 100}, checkpoint)
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            events=SimpleNamespace(
                reset_plug_curriculum=SimpleNamespace(params={"num_steps_per_env": 512}),
            )
        ),
        common_step_counter=0,
    )

    assert DisplayportInsertionEnv._restore_step_counter(env, str(checkpoint))

    assert env.common_step_counter == 101 * 512


def test_success_smoothing_weights_the_number_of_completed_episodes(tmp_path):
    """A large reset batch contributes once per episode rather than once per call."""
    scheduler = _make_scheduler(tmp_path)
    scheduler._state_path = None
    scheduler._min_steps_between = 10_000
    succeeded = torch.tensor([True] * 5 + [False] * 5)
    env = SimpleNamespace(
        episode_succeeded=succeeded,
        common_step_counter=0,
        max_episode_length=100,
    )

    scheduler(env, torch.arange(10))

    expected_weight = 1.0 - (1.0 - scheduler.smoothing) ** 10
    assert scheduler._success_rate == pytest.approx(expected_weight * 0.5)


def test_first_reset_after_restore_applies_ranges_without_scoring(tmp_path):
    """The forced post-restore reset must not consume the pre-resume scene as an episode."""
    scheduler = _make_scheduler(tmp_path)
    scheduler._state_path = None
    scheduler.level = 9
    scheduler._success_rate = 0.3
    scheduler._skip_next_update = True
    env = SimpleNamespace(
        episode_succeeded=torch.ones(4, dtype=torch.bool),
        common_step_counter=10_000,
        max_episode_length=100,
    )

    result = scheduler(env, torch.arange(4))

    assert result["level"] == 9
    assert result["success_rate"] == pytest.approx(0.3)
    assert not scheduler._skip_next_update


def test_load_training_state_forces_a_full_reset_after_restore():
    """Restored ranges are applied and resampled before the first resumed rollout."""
    calls = []
    scheduler = SimpleNamespace(
        restore_from_checkpoint=lambda path: calls.append(("restore", path)) or True,
        prepare_resume_reset=lambda: calls.append(("prepare", None)),
    )
    env = SimpleNamespace(
        curriculum_manager=SimpleNamespace(cfg={"adr": SimpleNamespace(func=scheduler)}),
        _restore_step_counter=lambda path: calls.append(("step", path)),
        reset=lambda: calls.append(("reset", None)),
        extras={"log": {"reset_only": 1.0}},
    )

    DisplayportInsertionEnv.load_training_state(env, "/tmp/model_100.pt")

    assert calls == [
        ("step", "/tmp/model_100.pt"),
        ("restore", "/tmp/model_100.pt"),
        ("prepare", None),
        ("reset", None),
    ]
    assert env.extras["log"] == {}


def test_load_training_state_resets_when_only_the_step_counter_restores():
    """Iteration-driven schedules are applied before rollout even without ADR state."""
    calls = []
    env = SimpleNamespace(
        curriculum_manager=None,
        _restore_step_counter=lambda path: calls.append(("step", path)) or True,
        reset=lambda: calls.append(("reset", None)),
        extras={"log": {}},
    )

    DisplayportInsertionEnv.load_training_state(env, "/tmp/model_100.pt")

    assert calls == [("step", "/tmp/model_100.pt"), ("reset", None)]


def test_resume_without_sidecar_suppresses_reset_scoring_and_restarts_spacing(tmp_path):
    """A missing ADR sidecar must not score the reset or allow immediate promotion."""
    scheduler = _make_scheduler(tmp_path, step=50_000)
    scheduler.level = 0
    scheduler._success_rate = 1.0
    scheduler.prepare_resume_reset()
    env = SimpleNamespace(
        episode_succeeded=torch.ones(4, dtype=torch.bool),
        common_step_counter=50_000,
        max_episode_length=100,
    )

    result = scheduler(env, torch.arange(4))

    assert result["level"] == 0
    assert result["success_rate"] == pytest.approx(1.0)
    assert scheduler._last_change_step == 50_000


def test_success_tracking_does_not_depend_on_metric_logging(monkeypatch):
    """Turning off metric logging must not make the ADR scheduler observe all failures."""
    env = object.__new__(DisplayportInsertionEnv)
    env._track_episode_success = True
    env._log_success_metrics = False
    env.episode_succeeded = torch.zeros(2, dtype=torch.bool)
    env.extras = {}
    env._compute_success = lambda: (
        torch.tensor([True, False]),
        torch.zeros(2),
        torch.zeros(2),
    )

    monkeypatch.setattr(
        ManagerBasedRLEnv,
        "step",
        lambda self, action: ({}, torch.zeros(2), torch.zeros(2), torch.zeros(2), self.extras),
    )

    DisplayportInsertionEnv.step(env, torch.zeros(2, 1))

    torch.testing.assert_close(env.episode_succeeded, torch.tensor([True, False]))
    assert "log" not in env.extras


def test_terminal_step_success_is_visible_to_curriculum(monkeypatch):
    """Success first reached on a done step is sticky before the parent reset scores it."""
    env = object.__new__(DisplayportInsertionEnv)
    env._log_success_metrics = True
    env.episode_succeeded = torch.zeros(2, dtype=torch.bool)
    env.extras = {"log": {}}
    env._compute_success = lambda: (
        torch.tensor([True, False]),
        torch.zeros(2),
        torch.zeros(2),
    )
    observed = []

    def parent_reset(self, env_ids):
        observed.append(self.episode_succeeded[env_ids].clone())
        self.extras["log"] = {}

    monkeypatch.setattr(ManagerBasedRLEnv, "_reset_idx", parent_reset)

    DisplayportInsertionEnv._reset_idx(env, torch.tensor([0]))

    torch.testing.assert_close(observed[0], torch.tensor([True]))
    assert not env.episode_succeeded[0]
    assert env.extras["log"]["Metrics/terminal_success_rate"] == 1.0
