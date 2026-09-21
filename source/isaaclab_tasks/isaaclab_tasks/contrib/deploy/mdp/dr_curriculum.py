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

import json
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.curriculums import modify_env_param
from isaaclab.managers import ManagerTermBase

STATE_FILENAME = "adr_state.json"
"""Name of the sidecar that carries the curriculum level alongside a run's checkpoints."""

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import CurriculumTermCfg


class SuccessDifficultyScheduler(ManagerTermBase):
    """Global difficulty level driven by the policy's episode success rate.

    The level advances by one when a smoothed success rate exceeds
    ``success_threshold`` and at least ``min_steps_between`` policy steps have elapsed
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
        self.min_steps_between: int = int(params.get("min_steps_between", 3000))
        self.demote: bool = bool(params.get("demote", False))
        self.smoothing: float = float(params.get("smoothing", 0.1))

        self.level: int = int(params.get("init_level", 0))
        self._success_rate: float = 0.0
        self._last_change_step: int = 0

        # RSL-RL checkpoints carry only network and optimizer state, so the level is
        # persisted next to them instead. Only rank 0 writes: each rank derives its own
        # timestamped log directory, and concurrent writers would race.
        log_dir = getattr(env.cfg, "log_dir", None)
        is_main_rank = int(os.getenv("RANK", "0")) == 0
        self._state_path = os.path.join(log_dir, STATE_FILENAME) if log_dir and is_main_rank else None
        self._write_state()

        if not hasattr(env, "episode_succeeded"):
            raise ValueError(
                "SuccessDifficultyScheduler requires the environment to expose an 'episode_succeeded'"
                " flag. Use DisplayportInsertionEnv or add an equivalent buffer."
            )

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

    def _write_state(self) -> None:
        """Persist the level next to the run's checkpoints. Never fatal to training."""
        if self._state_path is None:
            return
        try:
            os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
            with open(self._state_path, "w") as handle:
                json.dump(
                    {
                        "level": self.level,
                        "num_levels": self.num_levels,
                        "success_rate": self._success_rate,
                    },
                    handle,
                )
        except OSError as exc:
            print(f"[WARN] Could not write the ADR state file '{self._state_path}': {exc}")

    def restore_from_checkpoint(self, checkpoint_path: str) -> bool:
        """Restore the level from the sidecar beside ``checkpoint_path``.

        Without this a resumed run silently restarts the curriculum at level 0, which
        snaps every randomization range back to its easy endpoint while the policy keeps
        its hard-trained weights.

        Returns:
            Whether a state file was found and applied.
        """
        state_file = os.path.join(os.path.dirname(checkpoint_path), STATE_FILENAME)
        if not os.path.isfile(state_file):
            print(
                f"[WARN] No ADR state file beside '{checkpoint_path}'. The curriculum will restart at"
                f" level {self.level}. Pass env.dr.adr.init_level to set it explicitly."
            )
            return False
        try:
            with open(state_file) as handle:
                state = json.load(handle)
        except (OSError, ValueError) as exc:
            print(f"[WARN] Could not read the ADR state file '{state_file}': {exc}")
            return False

        saved_level = int(state.get("level", 0))
        saved_num_levels = int(state.get("num_levels", self.num_levels))
        if saved_num_levels != self.num_levels and saved_num_levels > 0:
            # Preserve how far through the curriculum the run had progressed rather than
            # the raw count, so num_levels can be changed between runs.
            saved_level = round(saved_level * self.num_levels / saved_num_levels)
        self.level = max(0, min(saved_level, self.num_levels))
        self._success_rate = float(state.get("success_rate", 0.0))
        # common_step_counter restarts on resume, so the guard restarts with it.
        self._last_change_step = 0
        print(f"[INFO] Restored ADR curriculum level {self.level}/{self.num_levels} from '{state_file}'.")
        self._write_state()
        return True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        num_levels: int = 50,
        success_threshold: float = 0.4,
        min_steps_between: int = 3000,
        init_level: int = 0,
        demote: bool = False,
        smoothing: float = 0.1,
    ) -> dict[str, float]:
        succeeded = env.episode_succeeded[env_ids]
        if succeeded.numel() > 0:
            batch_rate = succeeded.float().mean().item()
            self._success_rate += self.smoothing * (batch_rate - self._success_rate)

        steps_since_change = env.common_step_counter - self._last_change_step
        if steps_since_change >= self.min_steps_between:
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
