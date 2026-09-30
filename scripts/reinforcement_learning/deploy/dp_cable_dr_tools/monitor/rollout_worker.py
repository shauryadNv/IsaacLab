# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drain the rollout queue: evaluate the newest queued checkpoint of each run, one at a time.

Older queued checkpoints of a run are skipped once a newer one is queued, so the worker
always catches up to "how is this run doing now". Runs whose last rollout is oldest go first.
Runs in its own cron entry (separate lock) so a slow rollout never delays status ticks.
"""

import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parent
QUEUE = HOME / "rollout_queue.jsonl"
DATA = HOME / "data"
# Set by ../env.sh (tick.sh sources it); the defaults mirror env.sh.
WT = Path(os.environ.get("ISAACLAB_DIR", str(HOME.parents[4])))
CONDA = os.environ.get("CONDA_BIN", os.path.expanduser("~/miniconda3/bin/conda"))  # conda run applies Isaac Sim's hooks
CONDA_ENV = os.environ.get("CONDA_ENV", "isaaclab_ship")
ENV_PY = os.environ.get("ISAACLAB_PYTHON", str(Path(CONDA).parent.parent / "envs" / CONDA_ENV / "bin" / "python"))
TIMEOUT_S = 20 * 60
# Training overrides that change the network or the controller must also apply at eval (a
# state_dependent_std=false checkpoint does not load into the default policy, and a payload-compensated
# policy is judged without compensation otherwise). DR and curriculum overrides are not passed: eval is nominal.
EVAL_OVERRIDE_PREFIXES = ("agent.policy.", "env.actions.")


def load(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def main() -> int:
    lock = open(HOME / ".rollout.lock", "w")  # noqa: SIM115 - held (flock) until the process exits
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("rollout worker already running")
        return 0
    if not QUEUE.exists():
        return 0

    latest: dict[tuple, dict] = {}
    for line in QUEUE.read_text().splitlines():
        try:
            job = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (job["sweep"], job["name"])
        if job["iteration"] >= latest.get(key, {}).get("iteration", -1):
            latest[key] = job

    todo = []
    for (sweep, name), job in latest.items():
        run_dir = DATA / sweep / name
        done = load(run_dir / "rollouts.json", [])
        if any(r.get("iteration") == job["iteration"] for r in done):
            continue
        last_time = max((r.get("time", "") for r in done), default="")
        todo.append((last_time, job))
    todo.sort(key=lambda x: x[0])

    src = WT / "source"
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join(
        str(src / p) for p in ("isaaclab", "isaaclab_tasks", "isaaclab_assets", "isaaclab_rl", "isaaclab_physx")
    )
    env.pop("DISPLAY", None)

    for _, job in todo:
        run_dir = DATA / job["sweep"] / job["name"]
        ckpt = Path(job["checkpoint"])
        if not ckpt.exists():  # superseded by a newer checkpoint the monitor already fetched
            continue
        out_dir = run_dir / "rollouts"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / f"iter_{job['iteration']:05d}.log"
        overrides = [
            o
            for o in load(run_dir / "status.json", {}).get("overrides", "").split()
            if o.startswith(EVAL_OVERRIDE_PREFIXES)
        ]
        print(
            f"rollout {job['sweep']}/{job['name']} iter {job['iteration']} {' '.join(overrides)}".rstrip(), flush=True
        )
        try:
            with log_path.open("w") as logf:
                subprocess.run(
                    [
                        CONDA,
                        "run",
                        "--no-capture-output",
                        "-n",
                        CONDA_ENV,
                        ENV_PY,
                        str(HOME / "rollout.py"),
                        str(ckpt),
                        str(out_dir),
                        "--iteration",
                        str(job["iteration"]),
                        *overrides,
                    ],
                    cwd=WT,
                    env=env,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    timeout=TIMEOUT_S,
                )
        except subprocess.TimeoutExpired:
            pass
        result = None
        for line in log_path.read_text(errors="replace").splitlines():
            if line.startswith("ROLLOUT_RESULT "):
                result = json.loads(line[len("ROLLOUT_RESULT ") :])
        if result is None:
            result = {"iteration": job["iteration"], "error": "rollout failed; see log", "log": log_path.name}
        result["model_iteration"] = job.get("model_iteration", result.get("model_iteration"))
        result["common_step_counter"] = job.get("common_step_counter")
        from datetime import datetime, timezone

        result["time"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        done = load(run_dir / "rollouts.json", [])
        done.append(result)
        (run_dir / "rollouts.json").write_text(json.dumps(done, indent=1))
        print(f"  -> {result.get('success_rate', result.get('error'))}", flush=True)
        sys.path.insert(0, str(HOME))
        import build_dashboard

        build_dashboard.build()
    return 0


if __name__ == "__main__":
    sys.exit(main())
