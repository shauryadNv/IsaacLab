# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One monitoring tick over tracked osmo training sweeps.

For every run in every tracked sweep:
  * query its osmo status (JSON), and relaunch it if it died, resuming from its newest
    checkpoint (so the DR curriculum level resumes too, via adr_state.json);
  * save the latest log tail and parse the latest iteration's console metrics;
  * mirror its swift artifacts (tfevents, adr_state.json, newest model_*.pt) locally and
    turn the tfevents into metrics.json;
  * queue a local rollout video when a new checkpoint appears;
then rebuild the dashboard.

Run from cron (see README.md). Safe to run concurrently: a lock makes overlapping ticks exit.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HOME = Path(__file__).resolve().parent
DATA = HOME / "data"
SWEEPS_FILE = HOME / "sweeps.json"  # relative "path" entries resolve against the tools dir (HOME.parent)
# Dry run: do everything (status, logs, sync, metrics, dashboard) but never submit, cancel, or edit resources.
DRY_RUN = os.environ.get("MONITOR_DRY_RUN", "0") == "1" or "--dry-run" in sys.argv
#  # {"<sweep>": {"path": <dir with config.sh, manifest.tsv, runs.tsv, submit.sh>, "label": ..., "description": ...}}
MAX_ATTEMPTS = 4  # original + 3 relaunches, then give up and flag it
# When a relaunch goes out at HIGH priority (not preemptible; queues within the pool's HIGH/NORMAL quota).
# Per sweep via sweeps.json "relaunch_high": "last" = only the final allowed attempt, "always", or "never".
# A run's own resources/<name>.sh can pin PRIORITY=HIGH regardless.
RELAUNCH_HIGH_DEFAULT = "last"
DEAD = ("FAILED", "CANCELED", "TIMEOUT", "ERROR")  # prefixes; osmo also reports e.g. FAILED_PREEMPTED
DONE = ("COMPLETED",)
UA = {"User-Agent": "run-monitor"}
LOCK = threading.Lock()  # guards the shared queue file and runs.tsv appends across worker threads


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"{now()} {msg}", flush=True)


def read_config(sweep_dir: Path) -> dict:
    """The sweep's config.sh variables, evaluated by bash (so values computed from env.sh resolve)."""
    out = subprocess.run(
        ["bash", "-c", "set -a; source ./config.sh >/dev/null; env -0"],
        cwd=sweep_dir,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout
    return dict(kv.split("=", 1) for kv in out.split("\0") if "=" in kv)


def read_tsv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    head = lines[0].split("\t")
    return [dict(zip(head, line.split("\t"))) for line in lines[1:] if line.strip()]


def append_event(run_dir: Path, **event) -> None:
    with (run_dir / "events.jsonl").open("a") as f:
        f.write(json.dumps({"time": now(), **event}) + "\n")


# ------------------------------------------------------------------ osmo


def osmo(*args: str, timeout: int = 90) -> str:
    """Run the osmo CLI; on timeout return whatever it printed so far (log streams stay open until
    the server gives up, so a short timeout still yields the recent lines)."""
    try:
        return subprocess.run(["osmo", *args], capture_output=True, text=True, timeout=timeout).stdout
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        return out.decode("utf-8", "replace") if isinstance(out, bytes) else out


def is_preemption(q: dict) -> bool:
    """True if osmo reports the workflow was killed by the platform (exit 2001), not by the task."""
    msgs = " ".join((t.get("failure_message") or "") for g in (q.get("groups") or []) for t in (g.get("tasks") or []))
    return "2001" in msgs or "OSMO Control failure" in msgs


def osmo_query(rid: str, tries: int = 3) -> dict:
    """Workflow status as a dict; {} (shown as UNKNOWN, never relaunched) if osmo keeps failing."""
    for attempt in range(tries):
        out = osmo("workflow", "query", rid, "--format-type", "json")
        start = out.find("{")
        if start >= 0:
            try:
                return json.loads(out[start:])
            except json.JSONDecodeError:
                pass
        time.sleep(5 * (attempt + 1))  # transient osmo/API failures under parallel load
    return {}


LOG_PREFIX = re.compile(r"^\d{4}/\d\d/\d\d \d\d:\d\d:\d\d \[[^\]]+\] ?")
ITER_RE = re.compile(r"Learning iteration (\d+)/(\d+)")
KV_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_ /.\-()]{0,80}?):\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?![\d.])")


def parse_console(text: str) -> dict:
    """Latest iteration and its metric block from RSL-RL's console output."""
    lines = [LOG_PREFIX.sub("", ln) for ln in text.splitlines()]
    last = None
    for i, ln in enumerate(lines):
        m = ITER_RE.search(ln)
        if m:
            last = (i, int(m.group(1)), int(m.group(2)))
    if last is None:
        return {}
    i0, it, total = last
    metrics = {}
    for ln in lines[i0 + 1 : i0 + 80]:
        if ITER_RE.search(ln):
            break
        m = KV_RE.match(ln)
        if m:
            metrics[m.group(1).strip()] = float(m.group(2))
    return {"iteration": it, "max_iterations": total, "metrics": metrics}


# ------------------------------------------------------------------ swift


ROW_RE = re.compile(
    r'<tr class="item ([^"]*)">\s*<td class="colname"><a href="\./([^"]+)">.*?</a></td>\s*'
    r'<td class="colsize">([^<]*)</td>\s*<td class="coldate">([^<]*)</td>',
    re.S,
)


def swift_list(url: str) -> list[dict]:
    """Entries of a swift static-web listing: name, is_dir, size, date. [] if absent."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
            page = r.read().decode("utf-8", "replace")
    except Exception:
        return []
    return [
        {
            "name": urllib.parse.unquote(n).rstrip("/"),
            "is_dir": "subdir" in cls or n.endswith("/"),
            "size": sz.strip().replace("&nbsp;", ""),
            "date": d.strip().replace("&nbsp;", ""),
        }
        for cls, n, sz, d in ROW_RE.findall(page)
    ]


def download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r, tmp.open("wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.replace(dest)
        return True
    except Exception as exc:
        log(f"  download failed {url}: {exc}")
        tmp.unlink(missing_ok=True)
        return False


def model_iter(name: str) -> int:
    m = re.match(r"model_(\d+)\.pt$", name)
    return int(m.group(1)) if m else -1


def sync_attempt(cfg: dict, attempt: dict, run_dir: Path, manifest: dict) -> dict:
    """Mirror one attempt's swift folder. Returns its newest checkpoint info (or {})."""
    # Each attempt records the commit it ran (runs.tsv "commit"); relaunches may use a newer one.
    commit = attempt.get("commit") or cfg["COMMIT"]
    base = f"{cfg['SWIFT_HTTP']}/{commit}{attempt['run_tag']}/displayport_insertion_{cfg['ROBOT_TYPE']}/"
    newest = {}
    for folder in (e for e in swift_list(base) if e["is_dir"]):
        entries = swift_list(base + urllib.parse.quote(folder["name"]) + "/")
        local = run_dir / "artifacts" / f"a{attempt['attempt']}" / folder["name"]
        for e in entries:
            if e["is_dir"]:
                continue
            if e["name"].startswith("events.out.tfevents") or e["name"] == "adr_state.json":
                dest = local / e["name"]
                stamp = dest.with_suffix(dest.suffix + ".remote")
                sig = f"{e['size']}|{e['date']}"
                if not dest.exists() or not stamp.exists() or stamp.read_text() != sig:
                    if download(base + urllib.parse.quote(folder["name"]) + "/" + e["name"], dest):
                        stamp.write_text(sig)
        models = sorted((e for e in entries if model_iter(e["name"]) >= 0), key=lambda e: model_iter(e["name"]))
        if models and model_iter(models[-1]["name"]) > newest.get("iteration", -1):
            newest = {
                "iteration": model_iter(models[-1]["name"]),
                "file": models[-1]["name"],
                "folder": folder["name"],
                "url": base + urllib.parse.quote(folder["name"]) + "/" + models[-1]["name"],
                "resume_from": (
                    f"{commit}{attempt['run_tag']}/displayport_insertion_{cfg['ROBOT_TYPE']}/{folder['name']}"
                ),
                "attempt": int(attempt["attempt"]),
            }
    return newest


def fetch_latest_checkpoint(ckpt: dict, run_dir: Path) -> Path | None:
    """Keep only the newest checkpoint locally (checkpoints are ~20 MB each)."""
    dest = run_dir / "checkpoint" / ckpt["file"]
    if not dest.exists():
        if not download(ckpt["url"], dest):
            return None
        for old in dest.parent.glob("model_*.pt"):
            if old != dest:
                old.unlink()
        # the rollout needs the ADR sidecar next to the checkpoint only for training; copy for reference
    return dest


def build_metrics(run_dir: Path) -> None:
    """Merge every attempt's tfevents into metrics.json: {tag: [[step, value], ...]}."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return
    merged: dict[str, dict[int, float]] = {}
    for events in sorted((run_dir / "artifacts").glob("a*/*/events.out.tfevents*")):
        acc = EventAccumulator(str(events), size_guidance={"scalars": 0})
        try:
            acc.Reload()
        except Exception:
            continue
        for tag in acc.Tags().get("scalars", []):
            if tag.endswith("/time"):  # RSL-RL's wall-clock-indexed duplicates; not iteration-indexed
                continue
            series = merged.setdefault(tag, {})
            for ev in acc.Scalars(tag):
                series[ev.step] = ev.value  # later attempts overwrite overlapping steps
    out = {tag: sorted([s, v] for s, v in series.items()) for tag, series in merged.items()}
    (run_dir / "metrics.json").write_text(json.dumps(out))


# ------------------------------------------------------------------ relaunch


def relaunch(
    sweep_dir: Path,
    cfg: dict,
    name: str,
    attempts: list[dict],
    newest: dict,
    run_dir: Path,
    counted: int,
    high_policy: str,
) -> None:
    """Resubmit a dead run. ``attempts`` = every runs.tsv row (for numbering); ``counted`` = attempts that
    count toward MAX_ATTEMPTS, so ``counted + 1`` is the number this relaunch will be."""
    if DRY_RUN:
        log(f"  DRY-RUN: would relaunch {name} (dead run {attempts[-1]['run_id']})")
        return
    nxt = int(attempts[-1]["attempt"]) + 1
    env = dict(os.environ, ATTEMPT=str(nxt))
    is_last = counted + 1 >= MAX_ATTEMPTS
    high = high_policy == "always" or (high_policy == "last" and is_last)
    if high:
        env["PRIORITY"] = "HIGH"
    max_total = int(cfg.get("MAX_ITERATIONS", "1500"))
    if newest:
        remaining = max_total - newest["iteration"]
        if remaining <= 0:
            append_event(run_dir, kind="done", note=f"checkpoint {newest['iteration']} already reached {max_total}")
            return
        env.update(RESUME_FROM=newest["resume_from"], MAX_ITERS=str(remaining))
        note = f"resume from model_{newest['iteration']} ({remaining} iterations left)"
    else:
        note = "no checkpoint yet; restarting fresh"
    if high:
        note += f" at HIGH priority ({'final allowed attempt' if is_last else 'policy: always'})"
    out = subprocess.run(
        [str(sweep_dir / "submit.sh"), name, "relaunch"], env=env, capture_output=True, text=True, timeout=180
    )
    ok = out.returncode == 0
    log(
        f"  RELAUNCH {name} attempt {nxt}: {note} -> "
        f"{'ok ' + out.stdout.strip() if ok else 'FAILED ' + out.stderr.strip()}"
    )
    append_event(
        run_dir,
        kind="relaunch" if ok else "relaunch_failed",
        attempt=nxt,
        note=note,
        detail=(out.stdout or out.stderr).strip()[-300:],
    )


def oom_fallback(sweep_dir: Path, name: str, rid: str) -> bool:
    """Switch ``name`` to 4 GPU x 1024 envs via resources/<name>.sh. False if it already uses 4 GPUs."""
    if DRY_RUN:
        log(f"  DRY-RUN: would switch {name} to 4 GPU x 1024 envs after OOM on {rid}")
        return False
    path = sweep_dir / "resources" / f"{name}.sh"
    existing = path.read_text() if path.exists() else ""
    if re.search(r"^NUM_GPUS=4\b", existing, re.M):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    block = (
        f"\n# {now()} monitor: CUDA OOM on {rid} -> fall back to 4 GPU x 1024 envs (same total envs).\n"
        "NUM_GPUS=4\nNUM_ENVS=1024\nNUM_CPUS=40\nMEMORY=128Gi\n"
        "TAG_SUFFIX=${TAG_SUFFIX/g2e2048/g4e1024}\n"
    )
    path.write_text(existing + block)
    return True


# ------------------------------------------------------------------ tick


def tick_sweep(sweep: str, sweep_dir: Path, high_policy: str = RELAUNCH_HIGH_DEFAULT) -> None:
    cfg = read_config(sweep_dir)
    cfg["SWIFT_HTTP"] = cfg["SWIFT_ROOT"].replace("swift://pdx.s8k.io/", "https://pdx.s8k.io/v1/")
    manifest = {r["name"]: r["extra_overrides"] for r in read_tsv(sweep_dir / "manifest.tsv")}
    runs = read_tsv(sweep_dir / "runs.tsv")
    queued = set()
    qf = HOME / "rollout_queue.jsonl"
    if qf.exists():
        for line in qf.read_text().splitlines():
            try:
                j = json.loads(line)
                queued.add(f"{j['sweep']}/{j['name']}/{j['iteration']}")
            except (json.JSONDecodeError, KeyError):
                pass

    def one(name: str) -> None:
        all_attempts = [r for r in runs if r["name"] == name]
        # runs.tsv reason prefixes (numbering always uses every row):
        #   "superseded:" replaced by a later attempt; its artifacts are dropped and it does not count
        #                 toward MAX_ATTEMPTS (e.g. a buggy resume).
        #   "moved:"      cancelled on purpose and resubmitted elsewhere (e.g. to HIGH priority); its
        #                 artifacts and checkpoints are kept, but it does not count toward MAX_ATTEMPTS.
        data_attempts = [r for r in all_attempts if not r.get("reason", "").startswith("superseded:")]
        attempts = [r for r in data_attempts if not r.get("reason", "").startswith("moved:")]
        run_dir = DATA / sweep / name
        run_dir.mkdir(parents=True, exist_ok=True)
        status = {"sweep": sweep, "name": name, "overrides": manifest[name], "updated": now()}
        if not all_attempts:
            status["status"] = "NOT_SUBMITTED"
            (run_dir / "status.json").write_text(json.dumps(status, indent=1))
            return
        cur = all_attempts[-1]
        rid = cur["run_id"]
        q = osmo_query(rid)
        st = q.get("status", "UNKNOWN")
        prev = json.loads((run_dir / "status.json").read_text()) if (run_dir / "status.json").exists() else {}
        # Preemptions (osmo exit 2001 "OSMO Control failure") are not the run's fault: they do not count
        # toward MAX_ATTEMPTS and do not escalate priority. Recorded per attempt in preempted.json.
        pre_file = run_dir / "preempted.json"
        preempted = set(json.loads(pre_file.read_text())) if pre_file.exists() else set()
        if st.startswith(DEAD) and is_preemption(q) and rid not in preempted:
            preempted.add(rid)
            pre_file.write_text(json.dumps(sorted(preempted)))
        attempts = [r for r in attempts if r["run_id"] not in preempted]
        if prev.get("status") != st or prev.get("run_id") != rid:
            append_event(run_dir, kind="status", run_id=rid, status=st)
        status.update(
            run_id=rid,
            attempt=int(cur["attempt"]),
            attempts=len(attempts),
            status=st,
            priority=cur.get("priority") or "LOW",
            preemptions=len(preempted),
            overview=q.get("overview"),
            submit_time=q.get("submit_time"),
            start_time=q.get("start_time"),
            end_time=q.get("end_time"),
            cancelled_by=q.get("cancelled_by"),
            max_iterations=int(cfg.get("MAX_ITERATIONS", 1500)),
        )

        text = ""
        if st.startswith(DEAD) and st != prev.get("status"):
            # The stream prints oldest-first, so a capped fetch of a long tail loses the traceback at the end;
            # for a run that just died, fetch a short tail with a generous timeout.
            text = osmo("workflow", "logs", rid, "-n", "400", timeout=150)
        elif st == "RUNNING" or (st != prev.get("status") and st != "PENDING"):
            text = osmo("workflow", "logs", rid, "-n", "1500", timeout=45)
            if text.strip():
                (run_dir / "logs").mkdir(exist_ok=True)
                (run_dir / "logs" / f"{rid}.log").write_text(text)
                (run_dir / "logs" / "latest.log").write_text(text)
                console = parse_console(text)
                if console:
                    status["console"] = console

        newest = {}
        for a in data_attempts:
            info = sync_attempt(cfg, a, run_dir, manifest)
            if info.get("iteration", -1) > newest.get("iteration", -1):
                newest = info
        build_metrics(run_dir)
        if newest:
            status["checkpoint"] = newest
            local = fetch_latest_checkpoint(newest, run_dir)
            if local:
                status["checkpoint"]["local"] = str(local)
                done = (
                    json.loads((run_dir / "rollouts.json").read_text()) if (run_dir / "rollouts.json").exists() else []
                )
                key = f"{sweep}/{name}/{newest['iteration']}"
                if not any(r.get("iteration") == newest["iteration"] for r in done) and key not in queued:
                    with LOCK, (HOME / "rollout_queue.jsonl").open("a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "sweep": sweep,
                                    "name": name,
                                    "iteration": newest["iteration"],
                                    "checkpoint": str(local),
                                    "queued": now(),
                                }
                            )
                            + "\n"
                        )

        hold = (run_dir / "HOLD").exists()
        user_cancelled = bool(q.get("cancelled_by")) and q.get("cancelled_by") != "osmo"
        if not text and (run_dir / "logs" / f"{rid}.log").exists():
            text = (run_dir / "logs" / f"{rid}.log").read_text(errors="replace")
        # A CUDA OOM tends to repeat for a given config, so relaunching as-is burns GPU time. Rule (user,
        # 2026-09-25): OOM at 2 GPU x 2048 envs -> fall back to 4 GPU x 1024 envs, resuming from the newest
        # checkpoint; give up if it already runs on 4 GPUs. The flag is sticky per attempt because the saved
        # log tail can be partial on later ticks.
        # NCCL watchdog timeouts come from cluster/network stalls (several runs abort in the same minute), so
        # like preemptions they do not count toward MAX_ATTEMPTS.
        if st.startswith(DEAD) and "Watchdog caught collective operation timeout" in text and rid not in preempted:
            preempted.add(rid)
            pre_file.write_text(json.dumps(sorted(preempted)))
            attempts = [r for r in attempts if r["run_id"] not in preempted]
            status["attempts"] = len(attempts)
            status["preemptions"] = len(preempted)
        oom = st.startswith(DEAD) and (
            "OutOfMemoryError" in text
            or "CUDA out of memory" in text
            or bool(prev.get("oom") and prev.get("run_id") == rid)
        )
        status["oom"] = oom
        if oom and not hold and not user_cancelled:
            # Switching to 4 GPUs is a config change, not a blind retry: allowed once even at the attempt cap.
            if len(attempts) <= MAX_ATTEMPTS and oom_fallback(sweep_dir, name, rid):
                append_event(
                    run_dir,
                    kind="oom_fallback",
                    note=f"CUDA OOM on {rid}: resources/{name}.sh now 4 GPU x 1024 envs; relaunching",
                )
                with LOCK:
                    relaunch(sweep_dir, cfg, name, all_attempts, newest, run_dir, len(attempts), high_policy)
            else:
                status["gave_up"] = True
                if not prev.get("gave_up"):
                    append_event(run_dir, kind="gave_up", note="CUDA out of memory at 4 GPUs (or attempts used up)")
        elif oom:
            status["gave_up"] = True
        elif st.startswith(DEAD) and not hold and not user_cancelled:
            if len(attempts) < MAX_ATTEMPTS:
                with LOCK:
                    relaunch(sweep_dir, cfg, name, all_attempts, newest, run_dir, len(attempts), high_policy)
            else:
                status["gave_up"] = True
                if not prev.get("gave_up"):
                    append_event(run_dir, kind="gave_up", note=f"{len(attempts)} attempts failed")
        status["hold"] = hold
        status["user_cancelled"] = user_cancelled
        (run_dir / "status.json").write_text(json.dumps(status, indent=1))
        log(
            f"  {name:22s} {st:10s} attempt {cur['attempt']} "
            f"iter {status.get('console', {}).get('iteration', '-')} ckpt {newest.get('iteration', '-')}"
        )

    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(one, manifest))


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    lock = open(HOME / ".monitor.lock", "w")  # noqa: SIM115 - held (flock) until the process exits
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another tick is running; exiting")
        return 0
    sweeps = json.loads(SWEEPS_FILE.read_text())
    if DRY_RUN:
        log("DRY-RUN: no submits, cancels, or resource edits")
    for sweep, entry in sweeps.items():
        sweep_dir = entry["path"] if isinstance(entry, dict) else entry
        sweep_dir = sweep_dir if Path(sweep_dir).is_absolute() else str(HOME.parent / sweep_dir)
        high_policy = (
            entry.get("relaunch_high", RELAUNCH_HIGH_DEFAULT) if isinstance(entry, dict) else RELAUNCH_HIGH_DEFAULT
        )
        log(f"sweep {sweep}")
        tick_sweep(sweep, Path(sweep_dir), high_policy)
    sys.path.insert(0, str(HOME))
    import build_dashboard

    build_dashboard.build()
    log("tick done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
