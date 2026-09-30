# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based insertion env with task-success detection and logging.

Adds success metrics to ``extras["log"]`` for RSL-RL without changing the MDP
observation, action, reward, or termination logic.
"""

from __future__ import annotations

import torch
import warp as wp

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import combine_frame_transforms


def _keypoint_offsets_6d(device: torch.device) -> torch.Tensor:
    """Return the 7 unit keypoint offsets used by the keypoint reward."""
    corners = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], device=device, dtype=torch.float32)
    return torch.cat((corners, -corners[-3:]), dim=0)


class DisplayportInsertionEnv(ManagerBasedRLEnv):
    """Manager-based RL env that logs insertion success metrics during training.

    The following scalars are added to ``extras["log"]``:

    - ``Metrics/success_rate``: fraction of envs within the success threshold
    - ``Metrics/plug_socket_pos_error_m``: mean mate-point distance (m)
    - ``Metrics/plug_socket_keypoint_dist_m``: mean keypoint distance (m)
    - ``Metrics/terminal_success_rate``: success fraction at episode reset
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        # Domain randomization is expanded here rather than in the config's __post_init__
        # because Isaac Lab applies ``env.*`` CLI overrides after config construction.
        # This runs after those overrides and before the managers are built.
        expand_randomization = getattr(cfg, "apply_domain_randomization", None)
        if callable(expand_randomization):
            expand_randomization()

        # Allocated from the config before the base constructor runs, because that
        # constructor builds the curriculum manager, whose scheduler reads this flag.
        self.episode_succeeded = torch.zeros(cfg.scene.num_envs, dtype=torch.bool, device=cfg.sim.device)
        """Sticky per-episode success flag, cleared at reset.

        True once an environment has reached the success threshold at any point in the
        current episode. Read by the domain-randomization curriculum, which advances on
        whether the task was achieved rather than on whether it happened to be achieved
        in the final frame.
        """

        super().__init__(cfg, render_mode=render_mode, **kwargs)

        self._log_success_metrics: bool = bool(getattr(cfg, "log_success_metrics", True))
        curriculum_cfg = getattr(getattr(self, "curriculum_manager", None), "cfg", None)
        adr_cfg = (
            curriculum_cfg.get("adr") if isinstance(curriculum_cfg, dict) else getattr(curriculum_cfg, "adr", None)
        )
        # Success is scheduler state, not logging state. Keep tracking it when ADR is
        # active even if the user suppresses metric emission.
        self._track_episode_success: bool = adr_cfg is not None
        self._success_socket_asset: str = getattr(cfg, "success_socket_asset", "dp_socket")
        self._success_plug_asset: str = getattr(cfg, "success_plug_asset", "dp_plug")
        self._success_pos_threshold: float = float(getattr(cfg, "success_pos_threshold", 0.003))
        self._success_keypoint_scale: float = float(getattr(cfg, "success_keypoint_scale", 0.15))

        device = self.device
        self._success_socket_offset = torch.tensor(
            getattr(cfg, "success_socket_offset", [0.0, 0.0, 0.0]), device=device, dtype=torch.float32
        )
        self._success_plug_offset = torch.tensor(
            getattr(cfg, "success_plug_offset", [0.0, 0.0, 0.0]), device=device, dtype=torch.float32
        )
        self._success_plug_goal_rot_inv = torch.tensor(
            getattr(cfg, "success_plug_goal_rot_inv", [0.0, 0.0, 0.0, 1.0]), device=device, dtype=torch.float32
        )

        self._success_identity_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32).repeat(
            self.num_envs, 1
        )
        self._success_kp_offsets = _keypoint_offsets_6d(device) * self._success_keypoint_scale

    def load_training_state(self, checkpoint_path: str) -> None:
        """Restore environment-side training state that the RL checkpoint does not carry.

        RSL-RL checkpoints hold only the networks, the optimizer, and the iteration count,
        so a resumed run would otherwise restart the randomization curriculum at level 0
        while keeping hard-trained weights. Called by the training entry point after the
        checkpoint is loaded; a no-op when no curriculum is configured.

        Also restores the environment step counter, which a new process starts at 0, so
        schedules driven by it (such as the at-goal spawn anneal) continue from the
        resumed iteration instead of starting over.

        Args:
            checkpoint_path: Path to the checkpoint being resumed from.
        """
        step_restored = self._restore_step_counter(checkpoint_path)
        adr_restored = False
        scheduler = None
        manager = getattr(self, "curriculum_manager", None)
        if manager is not None and manager.cfg is not None:
            cfg = manager.cfg
            term_cfg = cfg.get("adr") if isinstance(cfg, dict) else getattr(cfg, "adr", None)
            scheduler = getattr(term_cfg, "func", None)
            restore = getattr(scheduler, "restore_from_checkpoint", None)
            if callable(restore):
                adr_restored = bool(restore(checkpoint_path))

        if step_restored or adr_restored:
            prepare_reset = getattr(scheduler, "prepare_resume_reset", None)
            if callable(prepare_reset):
                prepare_reset()
            # The environment was initially reset using level-zero ranges before the
            # checkpoint path was known. A full reset runs the curriculum interpolation
            # terms at the restored level, then resamples every live randomized value
            # before the resumed policy collects its first rollout.
            self.reset()
            # Reset-only diagnostics are not a training rollout and must not appear as
            # its metrics in the first logger update.
            self.extras.get("log", {}).clear()

    def save_training_state(self, checkpoint_path: str) -> None:
        """Persist environment-side state uniquely alongside an RSL-RL checkpoint.

        Args:
            checkpoint_path: Path of the checkpoint that was just written.
        """
        manager = getattr(self, "curriculum_manager", None)
        if manager is None or manager.cfg is None:
            return
        cfg = manager.cfg
        term_cfg = cfg.get("adr") if isinstance(cfg, dict) else getattr(cfg, "adr", None)
        save = getattr(getattr(term_cfg, "func", None), "save_for_checkpoint", None)
        if callable(save):
            save(checkpoint_path)

    def _restore_step_counter(self, checkpoint_path: str) -> bool:
        """Restore ``common_step_counter`` to the checkpoint's rollout boundary.

        RSL-RL stores zero-based iteration ``N`` after completing rollout ``N``, so the
        checkpoint represents ``(N + 1) * num_steps_per_env`` environment steps. No-op
        when the at-goal reset term does not provide ``num_steps_per_env``.

        Returns:
            Whether the step counter was restored.
        """
        term = getattr(self.cfg.events, "reset_plug_curriculum", None)
        steps_per_iter = int((term.params if term is not None else {}).get("num_steps_per_env", 0) or 0)
        if steps_per_iter <= 0:
            return False
        iteration = int(torch.load(checkpoint_path, map_location="cpu", weights_only=False).get("iter", 0))
        completed_rollouts = iteration + 1
        self.common_step_counter = completed_rollouts * steps_per_iter
        print(
            f"[INFO] Restored common_step_counter={self.common_step_counter} "
            f"({completed_rollouts} completed rollouts x {steps_per_iter} steps) from '{checkpoint_path}'."
        )
        return True

    def _compute_success(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-env success mask, mate-point distance, and keypoint distance."""
        socket = self.scene[self._success_socket_asset]
        plug = self.scene[self._success_plug_asset]

        socket_pos = wp.to_torch(socket.data.root_pos_w)
        socket_quat = wp.to_torch(socket.data.root_quat_w)
        plug_pos = wp.to_torch(plug.data.root_pos_w)
        plug_quat = wp.to_torch(plug.data.root_quat_w)

        n = self.num_envs
        socket_off = self._success_socket_offset.unsqueeze(0).expand(n, -1)
        plug_off = self._success_plug_offset.unsqueeze(0).expand(n, -1)
        plug_goal_rot_inv = self._success_plug_goal_rot_inv.unsqueeze(0).expand(n, -1)

        # Mate reference frames (same construction as keypoint_two_body_error)
        kp_pos_s, kp_quat_s = combine_frame_transforms(socket_pos, socket_quat, socket_off, self._success_identity_quat)
        kp_pos_p, kp_quat_p = combine_frame_transforms(plug_pos, plug_quat, plug_off, plug_goal_rot_inv)

        pos_error = torch.linalg.norm(kp_pos_p - kp_pos_s, dim=-1)

        k = self._success_kp_offsets.shape[0]
        offs_flat = self._success_kp_offsets.unsqueeze(0).expand(n, -1, -1).reshape(-1, 3)
        ident_flat = self._success_identity_quat.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4)

        kp_s = combine_frame_transforms(
            kp_pos_s.unsqueeze(1).expand(-1, k, -1).reshape(-1, 3),
            kp_quat_s.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4),
            offs_flat,
            ident_flat,
        )[0].reshape(n, k, 3)
        kp_p = combine_frame_transforms(
            kp_pos_p.unsqueeze(1).expand(-1, k, -1).reshape(-1, 3),
            kp_quat_p.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4),
            offs_flat,
            ident_flat,
        )[0].reshape(n, k, 3)
        keypoint_dist = torch.linalg.norm(kp_p - kp_s, dim=-1).mean(dim=-1)

        is_success = pos_error < self._success_pos_threshold
        return is_success, pos_error, keypoint_dist

    def step(self, action: torch.Tensor):
        obs_buf, reward_buf, terminated, time_outs, extras = super().step(action)
        track_success = getattr(self, "_track_episode_success", False)
        log_success = getattr(self, "_log_success_metrics", False)
        if track_success or log_success:
            is_success, pos_error, keypoint_dist = self._compute_success()
            self.episode_succeeded |= is_success
        if log_success:
            log = self.extras.setdefault("log", {})
            log["Metrics/success_rate"] = is_success.float().mean()
            log["Metrics/plug_socket_pos_error_m"] = pos_error.mean()
            log["Metrics/plug_socket_keypoint_dist_m"] = keypoint_dist.mean()
        return obs_buf, reward_buf, terminated, time_outs, self.extras

    def _reset_idx(self, env_ids):
        terminal_success = None
        track_success = getattr(self, "_track_episode_success", False)
        log_success = getattr(self, "_log_success_metrics", False)
        if track_success or log_success:
            is_success, _, _ = self._compute_success()
            # ``super()._reset_idx`` runs the curriculum before changing scene
            # state, so include success reached on this terminal physics step in
            # the sticky episode result that the curriculum consumes.
            self.episode_succeeded[env_ids] |= is_success[env_ids]
            if log_success:
                terminal_success = is_success[env_ids].float().mean()

        # The curriculum manager runs inside the parent reset and reads
        # ``episode_succeeded``, so it must stay valid until after this call.
        super()._reset_idx(env_ids)
        self.episode_succeeded[env_ids] = False

        if terminal_success is not None:
            self.extras["log"]["Metrics/terminal_success_rate"] = terminal_success
