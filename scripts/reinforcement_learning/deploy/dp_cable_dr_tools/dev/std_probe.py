# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Where does a policy's action std blow up? Roll out a checkpoint with stochastic actions (as in training)
from (A) near-goal spawns (plug already partly inserted) and (B) approach spawns, and record the policy std
per step, env and action dimension. Prints one "PROBE {...}" JSON line.

  CKPT=<model.pt> COMP=0|1 LABEL=<name> python std_probe.py
"""

import argparse
import importlib.metadata as metadata
import json
import os
import sys

import torch
import warp as wp

CKPT = os.environ["CKPT"]
COMP = os.environ.get("COMP", "0") == "1"
LABEL = os.environ.get("LABEL", os.path.basename(CKPT))
N_ENVS = int(os.environ.get("N_ENVS", "64"))
STEPS = int(os.environ.get("STEPS", "150"))
sys.argv = sys.argv[:1]

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

TASK = "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-ROS-Inference"
DIMS = ["x", "y", "z", "rx", "ry", "rz"]


def t(x):
    return x.torch if hasattr(x, "torch") else wp.to_torch(x)


def pct(x, q):
    return round(torch.quantile(x.float().flatten(), q).item(), 4)


def main():
    p = argparse.ArgumentParser()
    add_launcher_args(p)
    args = p.parse_args(["--viz", "none"])
    env_cfg, agent_cfg = resolve_task_config(TASK, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = N_ENVS
    env_cfg.episode_length_s = 20.0  # no time-outs inside a phase
    env_cfg.terminations.plug_dropped = None
    env_cfg.terminations.plug_orientation_exceeded = None
    if COMP:
        env_cfg.actions.arm_action.payload_gravity_compensation = True
    args.device = env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    with launch_simulation(env_cfg, args):
        env = gym.make(TASK, cfg=env_cfg)
        u = env.unwrapped
        term = u.event_manager.get_term_cfg("reset_plug_curriculum")
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(CKPT)
        actor = runner.alg.actor
        actor.eval()
        out = {"label": LABEL, "ckpt": CKPT, "comp": COMP, "phases": {}}
        for phase, prob in (("near_goal_spawn", 1.0), ("approach_spawn", 0.0)):
            term.params["at_goal_prob"] = prob
            term.params["at_goal_prob_final"] = prob
            stds, seated = [], []
            with torch.inference_mode():  # the RNN hidden state must be created and reset in the same mode
                obs, _ = env.reset()
                actor.reset(torch.ones(u.num_envs, dtype=torch.bool, device=u.device))
                for _ in range(STEPS):
                    a = actor(obs, stochastic_output=True)
                    stds.append(actor.output_std.clone())
                    obs, _, dones, _ = env.step(a)
                    actor.reset(dones)
                    _, err, _ = u._compute_success()
                    seated.append(err < u._success_pos_threshold)
            std = torch.stack(stds)  # (steps, envs, 6)
            seat = torch.stack(seated)  # (steps, envs)
            ph = {
                "std_median": pct(std, 0.5),
                "std_p90": pct(std, 0.9),
                "std_p99": pct(std, 0.99),
                "std_max": round(std.max().item(), 4),
                "frac_std_gt_10": round((std > 10).float().mean().item(), 4),
                "per_dim_median": {d: pct(std[..., i], 0.5) for i, d in enumerate(DIMS)},
                "per_dim_p99": {d: pct(std[..., i], 0.99) for i, d in enumerate(DIMS)},
                "seated_frac": round(seat.float().mean().item(), 3),
            }
            if seat.any() and (~seat).any():
                ph["std_median_when_seated"] = pct(std[seat], 0.5)
                ph["std_p99_when_seated"] = pct(std[seat], 0.99)
                ph["std_median_when_not_seated"] = pct(std[~seat], 0.5)
                ph["std_p99_when_not_seated"] = pct(std[~seat], 0.99)
            out["phases"][phase] = ph
        print("PROBE " + json.dumps(out), flush=True)
        env.close()


if __name__ == "__main__":
    main()
