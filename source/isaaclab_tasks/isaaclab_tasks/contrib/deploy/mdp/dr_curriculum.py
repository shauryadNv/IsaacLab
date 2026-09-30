# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success-driven automatic domain randomization for deployment tasks.

A single global difficulty level advances when the policy succeeds often enough, and
every randomization range is interpolated from its easy endpoint to its hard endpoint by
that level. This is the scheme used by ADEPT (50 levels, 0.4 success trigger) and by the
dextrah-unified codebase (same, plus a minimum-steps guard between changes).
"""

from __future__ import annotations

__all__ = ["SuccessDifficultyScheduler", "interpolate_range_fn"]

import contextlib
import json
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.curriculums import modify_env_param
from isaaclab.managers import ManagerTermBase

STATE_FILENAME = "adr_state.json"
"""Name of the latest-state sidecar used for live monitoring."""

CHECKPOINT_STATE_SUFFIX = ".adr_state.json"
"""Suffix for curriculum state paired with one RSL-RL checkpoint."""

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import CurriculumTermCfg


class SuccessDifficultyScheduler(ManagerTermBase):
    """Global difficulty level driven by the policy's episode success rate.

    The level advances by one when a smoothed success rate exceeds
    ``success_threshold`` and at least ``min_episodes_between`` episodes' worth of policy steps have elapsed
    since the last change. Success is read from the environment's sticky
    :attr:`~isaaclab_tasks.contrib.deploy.cable_insertion.insertion_env.DisplayportInsertionEnv.episode_succeeded`
    flag, so an episode counts if the plug was ever seated, not only if it happened to be
    seated in the final frame.

    The level is exposed as :attr:`difficulty_frac` in ``[0, 1]`` for the interpolation
    terms to consume.

    .. note::
        The level is rank-local under multi-GPU training. Synchronizing it would require
        a collective, and curriculum terms only run when an environment resets, which
        happens at different steps on different ranks -- a collective there would
        deadlock. Ranks follow near-identical success curves, so levels stay close, and
        any residual spread simply adds randomization diversity.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self.num_levels: int = int(params.get("num_levels", 50))
        self.success_threshold: float = float(params.get("success_threshold", 0.4))
        self.min_episodes_between: float = float(params.get("min_episodes_between", 5.0))
        # Resolved from the episode length on first use; not reliably available while the
        # environment is still constructing its managers.
        self._min_steps_between: int | None = None
        self.demote: bool = bool(params.get("demote", False))
        self.smoothing: float = float(params.get("smoothing", 0.1))
        self.at_goal_event: str | None = params.get("at_goal_event", "reset_plug_curriculum")
        self._spawned_at_goal: torch.Tensor | None = None
        self._spawned_at_goal_resolved = False

        self.level: int = int(params.get("init_level", 0))
        self._success_rate: float = 0.0
        self._last_change_step: int = 0
        self._skip_next_update: bool = False

        # RSL-RL checkpoints carry only network and optimizer state, so the level is
        # persisted next to them instead. Only rank 0 writes: each rank derives its own
        # timestamped log directory, and concurrent writers would race.
        log_dir = getattr(env.cfg, "log_dir", None)
        is_main_rank = int(os.getenv("RANK", "0")) == 0
        self._state_path = os.path.join(log_dir, STATE_FILENAME) if log_dir and is_main_rank else None
        self._write_state()

    def _at_goal_spawns(self, env: ManagerBasedRLEnv) -> torch.Tensor | None:
        """Mask of envs whose episode began with the plug already at the goal, if tracked.

        Those episodes start partially inserted and on-axis, so scoring them would inflate
        the success rate the curriculum advances on. Resolved on first use because the
        event manager must exist before its terms can be looked up.
        """
        if not self._spawned_at_goal_resolved:
            self._spawned_at_goal_resolved = True
            event_manager = getattr(env, "event_manager", None)
            if self.at_goal_event and event_manager is not None:
                try:
                    term = event_manager.get_term_cfg(self.at_goal_event).func
                    self._spawned_at_goal = getattr(term, "spawned_at_goal", None)
                except ValueError:
                    self._spawned_at_goal = None
        return self._spawned_at_goal

    @property
    def difficulty_frac(self) -> float:
        """Current level as a fraction of the maximum, in ``[0, 1]``."""
        return self.level / max(self.num_levels, 1)

    def get_state(self) -> torch.Tensor:
        """Return the scheduler state so it survives a checkpoint round-trip."""
        return torch.tensor(
            [float(self.level), self._success_rate, float(self._last_change_step)], device=self._env.device
        )

    def set_state(self, state: torch.Tensor) -> None:
        values = state.flatten().tolist()
        self.level = int(values[0])
        self._success_rate = float(values[1])
        self._last_change_step = int(values[2])

    def _state_dict(self, checkpoint_path: str | None = None) -> dict[str, int | float | str]:
        """Return serializable scheduler state for a latest or checkpoint sidecar."""
        state: dict[str, int | float | str] = {
            "version": 2,
            "level": self.level,
            "num_levels": self.num_levels,
            "success_rate": self._success_rate,
            "step": int(getattr(self._env, "common_step_counter", 0)),
        }
        if checkpoint_path is not None:
            state["checkpoint"] = os.path.basename(checkpoint_path)
        return state

    def _write_json_state(self, state_path: str, checkpoint_path: str | None = None) -> bool:
        """Write scheduler state atomically. Errors warn instead of stopping training."""
        temporary_path = f"{state_path}.tmp"
        try:
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            with open(temporary_path, "w") as handle:
                json.dump(self._state_dict(checkpoint_path), handle)
            os.replace(temporary_path, state_path)
            return True
        except OSError as exc:
            print(f"[WARN] Could not write the ADR state file '{state_path}': {exc}")
            with contextlib.suppress(OSError):
                os.remove(temporary_path)
            return False

    def _write_state(self) -> None:
        """Persist the latest level for monitoring. Never fatal to training."""
        if self._state_path is not None:
            self._write_json_state(self._state_path)

    def save_for_checkpoint(self, checkpoint_path: str) -> bool:
        """Persist state in a sidecar uniquely paired with ``checkpoint_path``."""
        if self._state_path is None:
            return False
        state_path = os.path.splitext(checkpoint_path)[0] + CHECKPOINT_STATE_SUFFIX
        written = self._write_json_state(state_path, checkpoint_path)
        if written:
            self._write_state()
        return written

    def prepare_resume_reset(self) -> None:
        """Prepare for the environment reset performed immediately after a resume.

        The reset exists only to apply restored step-driven schedules and ADR ranges; it
        is not a completed episode. Suppress that scheduler update and restart the
        promotion-spacing guard at the restored rollout boundary. This is required even
        when no ADR sidecar could be restored.
        """
        self._skip_next_update = True
        self._last_change_step = int(getattr(self._env, "common_step_counter", 0))

    def restore_from_checkpoint(self, checkpoint_path: str) -> bool:
        """Restore the level from the sidecar beside ``checkpoint_path``.

        Without this a resumed run silently restarts the curriculum at level 0, which
        snaps every randomization range back to its easy endpoint while the policy keeps
        its hard-trained weights.

        Returns:
            Whether a state file was found and applied.
        """
        state_file = os.path.splitext(checkpoint_path)[0] + CHECKPOINT_STATE_SUFFIX
        if not os.path.isfile(state_file):
            # New latest-state files carry their write step, which lets us prove that
            # they do not come from after the requested checkpoint. Legacy files have
            # no such identity and are unsafe for an older checkpoint.
            latest_state_file = os.path.join(os.path.dirname(checkpoint_path), STATE_FILENAME)
            if not os.path.isfile(latest_state_file):
                print(
                    f"[WARN] No ADR state file beside '{checkpoint_path}'. The curriculum will restart at"
                    f" level {self.level}. Pass env.dr.adr.init_level to set it explicitly."
                )
                return False
            state_file = latest_state_file
        try:
            with open(state_file) as handle:
                state = json.load(handle)
        except (OSError, ValueError) as exc:
            print(f"[WARN] Could not read the ADR state file '{state_file}': {exc}")
            return False

        expected_checkpoint = os.path.basename(checkpoint_path)
        saved_checkpoint = state.get("checkpoint")
        if saved_checkpoint is not None and saved_checkpoint != expected_checkpoint:
            print(
                f"[WARN] ADR state '{state_file}' belongs to '{saved_checkpoint}', not"
                f" '{expected_checkpoint}'. The curriculum will restart at level {self.level}."
            )
            return False
        saved_step = state.get("step")
        if saved_checkpoint is None:
            checkpoint_step = int(self._env.common_step_counter)
            # A mutable latest-state file is safe only when it identifies the exact
            # rollout boundary represented by the checkpoint. Reject both future and
            # stale state; either can silently apply the wrong curriculum level or EMA.
            if saved_step is None or int(saved_step) != checkpoint_step:
                print(
                    f"[WARN] Latest ADR state '{state_file}' cannot be safely matched to"
                    f" '{expected_checkpoint}'. The curriculum will restart at level {self.level}."
                )
                return False
        elif saved_step is not None:
            # RSL-RL's zero-based iteration can repeat after a resume, so the checkpoint
            # field alone cannot reconstruct cumulative environment steps across multiple
            # preemptions. The exact-paired sidecar is authoritative.
            checkpoint_step = int(saved_step)
            if checkpoint_step < 0:
                print(f"[WARN] ADR state '{state_file}' has an invalid negative step; ignoring it.")
                return False
            self._env.common_step_counter = checkpoint_step

        saved_level = int(state.get("level", 0))
        saved_num_levels = int(state.get("num_levels", self.num_levels))
        if saved_num_levels != self.num_levels and saved_num_levels > 0:
            # Preserve how far through the curriculum the run had progressed rather than
            # the raw count, so num_levels can be changed between runs.
            saved_level = round(saved_level * self.num_levels / saved_num_levels)
        self.level = max(0, min(saved_level, self.num_levels))
        self._success_rate = float(state.get("success_rate", 0.0))
        self.prepare_resume_reset()
        print(f"[INFO] Restored ADR curriculum level {self.level}/{self.num_levels} from '{state_file}'.")
        self._write_state()
        return True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        num_levels: int = 50,
        success_threshold: float = 0.4,
        min_episodes_between: float = 5.0,
        init_level: int = 0,
        demote: bool = False,
        smoothing: float = 0.1,
        at_goal_event: str | None = "reset_plug_curriculum",
    ) -> dict[str, float]:
        # Checked here rather than in the constructor: the environment builds its
        # managers before its own buffers exist.
        episode_succeeded = getattr(env, "episode_succeeded", None)
        if episode_succeeded is None:
            raise ValueError(
                "SuccessDifficultyScheduler requires the environment to expose an 'episode_succeeded'"
                " flag. Use DisplayportInsertionEnv or add an equivalent buffer."
            )
        if self._skip_next_update:
            self._skip_next_update = False
        else:
            succeeded = episode_succeeded[env_ids]
            spawned_at_goal = self._at_goal_spawns(env)
            if spawned_at_goal is not None:
                # Score only episodes that began at the approach pose.
                succeeded = succeeded[~spawned_at_goal[env_ids]]
            if succeeded.numel() > 0:
                batch_rate = succeeded.float().mean().item()
                # ``smoothing`` is the contribution of one completed episode. Using
                # the equivalent batch coefficient prevents one reset call containing
                # hundreds of episodes from weighing the same as a one-env reset.
                batch_smoothing = 1.0 - (1.0 - self.smoothing) ** succeeded.numel()
                self._success_rate += batch_smoothing * (batch_rate - self._success_rate)

            steps_since_change = env.common_step_counter - self._last_change_step
            if self._min_steps_between is None:
                self._min_steps_between = int(round(self.min_episodes_between * env.max_episode_length))
            if steps_since_change >= self._min_steps_between:
                changed = False
                if self._success_rate > self.success_threshold and self.level < self.num_levels:
                    self.level += 1
                    changed = True
                elif self.demote and self._success_rate < self.success_threshold and self.level > 0:
                    self.level -= 1
                    changed = True
                if changed:
                    self._last_change_step = env.common_step_counter
                    # At most num_levels writes per run, so the cost is negligible.
                    self._write_state()

        return {"level": float(self.level), "frac": self.difficulty_frac, "success_rate": self._success_rate}


def interpolate_range_fn(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    data,
    initial_value,
    final_value,
    difficulty_term_str: str = "adr",
):
    """Interpolate a config value from ``initial_value`` to ``final_value`` by difficulty.

    Handles arbitrarily nested lists and tuples, interpolating at the scalar leaves and
    preserving the original container types so downstream code sees the shape it expects.

    Args:
        data: The value currently held at the target address. Only its structure is used.
        initial_value: Value at difficulty 0.
        final_value: Value at the maximum difficulty.
        difficulty_term_str: Name of the :class:`SuccessDifficultyScheduler` curriculum term.
    """
    manager_cfg = env.curriculum_manager.cfg
    term_cfg = (
        manager_cfg[difficulty_term_str] if isinstance(manager_cfg, dict) else getattr(manager_cfg, difficulty_term_str)
    )
    scheduler: SuccessDifficultyScheduler = term_cfg.func
    frac = scheduler.difficulty_frac
    new_value = _lerp_nested(initial_value, final_value, frac)
    if new_value == data:
        return modify_env_param.NO_CHANGE
    return new_value


def _lerp_nested(initial, final, frac: float):
    if isinstance(initial, dict):
        return {key: _lerp_nested(value, final[key], frac) for key, value in initial.items()}
    if isinstance(initial, (list, tuple)):
        interpolated = [_lerp_nested(i, f, frac) for i, f in zip(initial, final)]
        return type(initial)(interpolated)
    value = initial + frac * (final - initial)
    return int(round(value)) if isinstance(initial, int) and isinstance(final, int) else value
