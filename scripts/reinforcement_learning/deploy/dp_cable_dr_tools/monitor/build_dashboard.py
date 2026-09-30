# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect every tracked run's status, metrics, rollouts and events into the files the
dashboard pages read:

  summary.json                         one entry per run (overview table + sparklines)
  data/<sweep>/<name>/metrics_small.json   every tag, downsampled to <= MAX_POINTS
"""

import contextlib
import json
from datetime import datetime, timezone
from pathlib import Path

HOME = Path(__file__).resolve().parent
DATA = HOME / "data"
MAX_POINTS = 400
SPARK_TAGS = ["Metrics/terminal_success_rate", "Metrics/success_rate", "Curriculum/adr/level", "Train/mean_reward"]


def load(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def downsample(series: list, n: int = MAX_POINTS) -> list:
    """Bucket-mean to <= n points, always keeping the last point exact."""
    if len(series) <= n:
        return series
    size = len(series) / n
    out = []
    for i in range(n):
        chunk = series[int(i * size) : int((i + 1) * size)] or [series[-1]]
        out.append([chunk[-1][0], sum(v for _, v in chunk) / len(chunk)])
    out[-1] = series[-1]
    return out


def last(series: list):
    return series[-1][1] if series else None


def sweep_meta() -> dict:
    """{sweep: {"label", "description"}} from sweeps.json (plain path entries get the sweep name)."""
    meta = {}
    for sweep, entry in load(HOME / "sweeps.json", {}).items():
        entry = entry if isinstance(entry, dict) else {}
        meta[sweep] = {
            "label": entry.get("label", sweep),
            "description": entry.get("description", ""),
            "tags": entry.get("tags", {}),
        }
    return meta


def build() -> None:
    meta = sweep_meta()
    runs = []
    for status_path in sorted(DATA.glob("*/*/status.json")):
        run_dir = status_path.parent
        st = load(status_path, {})
        metrics = load(run_dir / "metrics.json", {})
        small = {tag: downsample(s) for tag, s in metrics.items()}
        (run_dir / "metrics_small.json").write_text(json.dumps(small))
        rollouts = load(run_dir / "rollouts.json", [])
        good = [r for r in rollouts if "success_rate" in r]
        events = []
        if (run_dir / "events.jsonl").exists():
            for line in (run_dir / "events.jsonl").read_text().splitlines()[-200:]:
                with contextlib.suppress(json.JSONDecodeError):
                    events.append(json.loads(line))
        tb_iter = max((s[-1][0] for s in metrics.values() if s), default=None)
        console = st.get("console", {})
        runs.append(
            {
                **{
                    k: st.get(k)
                    for k in (
                        "sweep",
                        "name",
                        "overrides",
                        "run_id",
                        "attempt",
                        "attempts",
                        "status",
                        "submit_time",
                        "start_time",
                        "end_time",
                        "updated",
                        "gave_up",
                        "hold",
                        "user_cancelled",
                        "max_iterations",
                        "oom",
                        "priority",
                    )
                },
                "sweep_label": meta.get(st.get("sweep"), {}).get("label", st.get("sweep")),
                "sweep_description": meta.get(st.get("sweep"), {}).get("description", ""),
                "sweep_tags": meta.get(st.get("sweep"), {}).get("tags", {}),
                "path": str(run_dir.relative_to(HOME)),
                "iteration": max(v for v in (tb_iter, console.get("iteration")) if v is not None)
                if (tb_iter is not None or console.get("iteration") is not None)
                else None,
                "checkpoint_iter": (st.get("checkpoint") or {}).get("iteration"),
                "checkpoint_model_iter": (st.get("checkpoint") or {}).get("model_iteration"),
                "success_rate": last(metrics.get("Metrics/success_rate", [])),
                "terminal_success_rate": last(metrics.get("Metrics/terminal_success_rate", [])),
                "adr_level": last(metrics.get("Curriculum/adr/level", [])),
                "adr_success": last(metrics.get("Curriculum/adr/success_rate", [])),
                "mean_reward": last(metrics.get("Train/mean_reward", [])),
                "pos_err_mm": (last(metrics.get("Metrics/plug_socket_pos_error_m", [])) or 0) * 1000
                if metrics.get("Metrics/plug_socket_pos_error_m")
                else None,
                "console_metrics": console.get("metrics", {}),
                "spark": {t: downsample(metrics.get(t, []), 60) for t in SPARK_TAGS},
                "eval": good[-1] if good else None,
                "eval_history": [[r["iteration"], r["success_rate"]] for r in good],
                "rollouts": rollouts,
                "events": events,
                "tags": sorted(metrics),
                "has_log": (run_dir / "logs" / "latest.log").exists(),
            }
        )
    summary = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "sweeps": meta, "runs": runs}
    tmp = HOME / "summary.json.tmp"
    tmp.write_text(json.dumps(summary))
    tmp.replace(HOME / "summary.json")


if __name__ == "__main__":
    build()
