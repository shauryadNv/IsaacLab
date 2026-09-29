# Training run monitor

Tracks osmo training sweeps: status, auto-relaunch of dead runs (resuming from the newest
checkpoint), logs, TensorBoard metrics, and a local rollout video + eval success for every new
checkpoint. Everything is saved under `data/` and shown on a local dashboard.

## View the dashboard

The server listens on `127.0.0.1:8765` (started automatically by every tick). From your laptop:

    ssh -L 8765:localhost:8765 <this machine>
    # then open http://localhost:8765

- **Overview** (`index.html`): every run with status, progress, train success, eval success, ADR
  level, reward, error. Sort by any column. Tick up to 3 runs to overlay them, and pick any
  metric for a small-multiples grid across all runs.
- **Run page** (click a run): latest rollout video (plus every earlier one), eval success vs
  iteration, key training curves, a metric explorer, events (status changes, relaunches), Hydra
  overrides, and the log tail.

## Scheduling

One tick = `tick.sh`: makes sure the server is up, runs `monitor.py`, then starts
`rollout_worker.py` in the background. Install it in cron (every 15 min):

    (crontab -l 2>/dev/null; echo "*/15 * * * * $PWD/tick.sh") | crontab -   # run from this directory

Ticks overlap safely: a tick that starts while another is running exits
("another tick is running" in `logs/tick.log`).

## What a tick does (`monitor.py`)

For every run in each sweep listed in `sweeps.json`:

1. `osmo workflow query` for status. On a status change or while running, saves the log tail to
   `data/<sweep>/<run>/logs/` and parses the latest "Learning iteration" block.
2. Mirrors the run's swift folder over HTTPS: tfevents, `adr_state.json`, and only the newest
   `model_*.pt` (to `checkpoint/`). Builds `metrics.json` from all attempts' tfevents.
3. Queues a rollout when a new checkpoint appears.
4. If the run died (FAILED*/CANCELED*/TIMEOUT/ERROR) it relaunches via the sweep's `submit.sh`
   with `ATTEMPT=k+1 RESUME_FROM=<newest checkpoint folder> MAX_ITERS=<remaining>`, so the
   curriculum level resumes too. Not relaunched: runs cancelled by a person, runs with a `HOLD`
   file in their data dir, CUDA OOMs (deterministic - change the config instead), and runs that
   already used 4 attempts (shown as "Failed (gave up)").

`rollout_worker.py` evaluates the newest queued checkpoint of each run (older ones are skipped),
one at a time on the local GPU: `rollout.py` loads the checkpoint into the nominal task (DR off,
no at-goal spawns), records a close-up of env 0 for one episode, and scores 16 envs.

## Files

| Path | What |
|---|---|
| `sweeps.json` | sweeps to track: `{"name": "<dir with config.sh, manifest.tsv, runs.tsv, submit.sh>"}` |
| `data/<sweep>/<run>/status.json` | latest status, console metrics, checkpoint |
| `data/<sweep>/<run>/events.jsonl` | status changes, relaunches, give-ups |
| `data/<sweep>/<run>/metrics.json` | all TensorBoard scalars (`metrics_small.json` = downsampled) |
| `data/<sweep>/<run>/rollouts/` | `iter_NNNNN.mp4` + log per rollout; results in `rollouts.json` |
| `logs/tick.log`, `logs/rollout.log` | scheduler output |

To stop relaunching one run: `touch data/<sweep>/<run>/HOLD`.

## Priority and attempt bookkeeping

- Runs are submitted at `PRIORITY` from the sweep's `config.sh` (LOW: preemptible, borrows idle pool GPUs
  beyond the 32-GPU quota). `resources/<name>.sh` can pin a run to HIGH (all_on / all_off are).
- Relaunches follow `relaunch_high` in `sweeps.json`: `last` (default) = the final allowed attempt goes
  out at HIGH so it cannot be preempted, `always`, or `never`. HIGH runs queue until the quota has room
  and show a HIGH tag on the dashboard. `runs.tsv` records each attempt's `priority` and `commit`.
- Preemptions (osmo exit 2001) and NCCL watchdog timeouts are infrastructure failures: recorded per run in
  `data/<sweep>/<run>/preempted.json`, they do not count toward the 4-attempt cap and never escalate priority.
- `runs.tsv` reason prefixes: `superseded:` = replaced (artifacts dropped, not counted toward the
  4-attempt cap); `moved:` = cancelled on purpose and resubmitted (artifacts/checkpoints kept, not counted).
To track another sweep: add it to `sweeps.json` (its dir needs the same `config.sh` / `submit.sh`
/ `manifest.tsv` / `runs.tsv` layout as `../sweeps/dr_sweep`). Relative paths in `sweeps.json` resolve against `dp_cable_dr_tools/`.

Dry run (no relaunches, no OOM fallbacks, still mirrors status/metrics): `MONITOR_DRY_RUN=1 ./tick.sh` or
`python monitor.py --dry-run` after `source ../env.sh`.
