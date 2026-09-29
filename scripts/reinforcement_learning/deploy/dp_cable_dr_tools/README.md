# DP cable DR tools

Tooling for the domain-randomization (DR) work on the DisplayPort task-space insertion env
(`IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace`, branch `shauryad/dp_cable_dr`):
osmo sweep launchers, a run monitor that auto-relaunches dead runs and evaluates every new checkpoint,
a local dashboard, and the local test scripts used to verify the DR knobs and controller changes.

**Start with [HANDOFF.md](HANDOFF.md)** for what was built, what is running, and what was found.
This file is the runbook.

| Path | What |
|---|---|
| `env.sh` | Machine-specific settings (Isaac Lab checkout, conda env, osmo workflow, swift root, results dir). Every script sources it. |
| `osmo/dp_cable.yaml` | osmo training workflow (clones the fork at `commit_hash`, trains, uploads to swift, optional `resume_from`). |
| `sweeps/<sweep>/` | One sweep = `config.sh` (commit, pool, GPUs, envs, tag), `manifest.tsv` (variant -> Hydra overrides), `submit.sh`, `resources/<variant>.sh` (per-run overrides), `runs.tsv` (append-only submission log). |
| `monitor/` | Run monitor + dashboard. `tick.sh` (cron entry), `monitor.py`, `rollout_worker.py` + `rollout.py` (local eval), `build_dashboard.py`, `index.html` / `run.html`, `sweeps.json`. See `monitor/README.md`. |
| `dev/` | Local sim tests (knob read-back, per-knob sweeps with videos, payload compensation, gain searches, std probe). |
| `docs/findings/` | Write-ups of every local experiment. `docs/isaaclab_ship_newton_rollback.txt`: conda env package pins. |

Runtime state (`monitor/data/`, `monitor/logs/`, `monitor/summary.json`, `results/`) is git-ignored.

## 1. One-time setup

1. **Isaac Lab checkout** on `shauryad/dp_cable_dr` (this repo). `env.sh` finds it from its own location.
2. **Conda env with Isaac Sim** that can run this checkout. Used only for local sims (rollout eval, `dev/`).
   The original machine uses `isaaclab_ship` (Isaac Sim 5.x builds, torch 2.10+cu128, rsl-rl-lib 5.0.1,
   newton 1.6.0rc1, warp-lang 1.17.0, **moviepy<2**: Isaac Lab imports `moviepy.editor`). Scripts put this
   checkout's `source/*` first on `PYTHONPATH`, so the env's own Isaac Lab copy does not matter.
   Quick check: `source env.sh && /bin/bash -c '$CONDA_RUN $ISAACLAB_PYTHON -c "import isaaclab, rsl_rl"'`.
3. **Override `env.sh` defaults if your machine differs**, by exporting before running, e.g. in `~/.bashrc`:

       export CONDA_BIN=$HOME/miniconda3/bin/conda CONDA_ENV=my_isaaclab_env
       export SWIFT_ROOT_BASE=swift://pdx.s8k.io/AUTH_team-isaac/datasets/<you>

4. **osmo**: `osmo login` on the machine (CLI at `/usr/local/bin/osmo`), access to pool
   `isaac-dev-l40-04`, and the osmo credentials the workflow mounts: `gitlab_cred`, `github_cred`
   (PAT that can clone the fork), `omni-auth` (`omni_user` / `omni_pass`).
5. **Workflow targets** (`osmo/dp_cable.yaml`): it clones `https://github.com/shauryadNv/IsaacLab.git`
   and uploads to / resumes from `.../datasets/shauryad/displayport_insertion_<robot>/`. A commit must be
   **pushed to that fork** before a run can use it. If you use your own fork or swift folder, edit the
   `git clone` line, `RESUME_URL`, and the output `url:` in the yaml, and set `SWIFT_ROOT_BASE` to match
   (the monitor mirrors artifacts from `$SWIFT_ROOT_BASE/displayport_insertion_<robot>/`, anonymous HTTPS).

## 2. Launch a sweep

    cd sweeps
    cp -r dr_sweep_comp my_sweep && cd my_sweep && : > runs.tsv   # start with an empty submission log
    # edit config.sh: COMMIT (pushed!), TAG_SUFFIX (unique -> own swift folders), PRIORITY, NUM_GPUS/NUM_ENVS
    # edit manifest.tsv: one row per variant, "<name>\t<hydra overrides>"
    ./submit.sh --dry-run all_off           # renders the workflow, submits nothing
    ./submit.sh all_off initial             # -> "all_off -> isaaclab_train_rsl_rl-NNNN (LOW)", appended to runs.tsv
    for n in $(tail -n +2 manifest.tsv | cut -f1); do ./submit.sh "$n" initial; done

- `resources/<variant>.sh` overrides config for one variant (e.g. `PRIORITY=HIGH`, or
  `NUM_GPUS=4 NUM_ENVS=1024 TAG_SUFFIX=${TAG_SUFFIX/g2e2048/g4e1024}` after a CUDA OOM).
- Resume by hand (what the monitor does):
  `ATTEMPT=2 RESUME_FROM=<commit><run_tag>/displayport_insertion_rizon4s/<timestamp> MAX_ITERS=<left> ./submit.sh <variant> relaunch`.
  `RESUME_FROM` is the swift folder holding `model_*.pt` + `adr_state.json` (for 2-GPU runs, the rank-0 folder,
  the one with checkpoints). Use a commit >= 794f5d5909b or the resumed run collapses (HANDOFF.md, issue 9).
- Then register the sweep with the monitor: add to `monitor/sweeps.json`

      "my_sweep": {"path": "sweeps/my_sweep", "label": "short label", "description": "...",
                   "tags": {"Curriculum": "near-goal OFF", "Payload comp": "on", "Policy std": "global", "Gain DR": "capped"},
                   "relaunch_high": "last"}

## 3. Monitor and dashboard

    source env.sh
    monitor/tick.sh                                     # one tick: server up, monitor.py, rollout worker in background
    MONITOR_DRY_RUN=1 monitor/tick.sh                   # same, but never submits / cancels / edits resources
    (crontab -l 2>/dev/null; echo "*/15 * * * * $DR_TOOLS_HOME/monitor/tick.sh") | crontab -   # install
    ssh -L 8765:localhost:8765 <this machine>           # then open http://localhost:8765

A tick, per run in every sweep in `monitor/sweeps.json`:
- queries osmo status; saves log tails and parses the latest learning iteration;
- mirrors the run's swift folder (tfevents, `adr_state.json`, newest `model_*.pt`) into `monitor/data/`;
- queues a local rollout (`rollout.py`: nominal task, DR off, no at-goal spawns, 16 envs, one video) for each new
  checkpoint. The run's `agent.policy.*` / `env.actions.*` overrides are applied, so payload compensation and the
  std head match training;
- relaunches dead runs from the newest checkpoint (`ATTEMPT=k+1`, run tag `-r<k>`, remaining iterations).

Relaunch rules:

| Case | Action |
|---|---|
| FAILED / CANCELED (by osmo) / TIMEOUT / ERROR | relaunch from newest checkpoint; max 4 counted attempts, then "gave up" |
| osmo preemption (exit 2001 "OSMO Control failure") or NCCL watchdog timeout | relaunch, **not counted** (`data/<sweep>/<run>/preempted.json`) |
| CUDA OOM | once: write `resources/<variant>.sh` = 4 GPU x 1024 envs and relaunch; sticky OOM flag |
| cancelled by a person | never relaunched |
| `touch monitor/data/<sweep>/<run>/HOLD` | never relaunched |
| last counted attempt | submitted at HIGH (`relaunch_high: last`; also `always` / `never`) |

`runs.tsv` reason prefixes: `superseded:` = attempt replaced, its artifacts ignored; `moved:` = cancelled on purpose
and resubmitted (e.g. LOW -> HIGH), artifacts kept. Neither counts toward the attempt cap.

## 4. Local tests (`dev/`)

All scripts source `../env.sh` and write to `$DR_TOOLS_RESULTS/<experiment>/` (default `./results/`). Headless.

| Script | What | Time |
|---|---|---|
| `dr_verify_live.sh` | every knob's sampled values read back from the live sim at ADR 0 and 50 | ~5 min |
| `dr_knob_sweep.sh [knob ...]` | one knob at a time, ADR 0/25/50, 32 envs: range checks + effect + side-by-side videos. `SWEEP_COMP=1` payload comp, `SWEEP_EXCITE=insert` scripted insertion, `SWEEP_OUT=` | ~3 min/knob |
| `dr_knob_sweep_report.py <results.jsonl>` | table from a knob sweep | s |
| `payload_comp_check.py` | hold / push / yaw / insertion tests. Env vars: `COMP=0/1`, `TESTS`, `STIFF` / `DAMP` (6 comma-separated gains), `DECOUPLE`, `JFRIC`, `INSERT_CLAMP`, `N_ENVS`, `VIDEO`, `LABEL` | ~2 min |
| `payload_comp_videos.sh` | comp off vs on videos | ~5 min |
| `gain_sweep.sh`, `decoupled_*.sh`, `osc_cap_*.sh` | the gain / decoupling / OSC-DR-cap searches in `docs/findings/` | 10-60 min |
| `dr_slip_and_wrench.sh` | finger-friction slip test + plug wrench demo video | ~5 min |
| `dr_visual_test.sh {gui|video}` | short training run with DR on, to look at it | - |
| `CKPT=<model.pt> COMP=0/1 std_probe.py` | per-state policy std distribution of a checkpoint (near-goal vs approach spawns) | ~2 min |

Evaluate one checkpoint by hand:

    source env.sh && cd $ISAACLAB_DIR && /bin/bash -c '$CONDA_RUN $ISAACLAB_PYTHON \
      $DR_TOOLS_HOME/monitor/rollout.py <model_N.pt> <out_dir> [--no-video] [env.actions.arm_action.payload_gravity_compensation=true]'

## 5. Taking over the live instance

The live instance on the original machine is `~/workspaces/rl_policy/dr_tools/` (cron:
`*/15 * * * * /home/shauryad/workspaces/rl_policy/run_monitor/tick.sh`, `run_monitor -> dr_tools/monitor`).
This directory is a copy of its code with paths parameterized. Its `sweeps/*/runs.tsv` is a snapshot taken
2026-09-29; the live files keep growing. To move the monitor here without losing history:

    crontab -e                                   # remove the old tick line first (two monitors would double-relaunch)
    rsync -a ~/workspaces/rl_policy/dr_tools/sweeps/ sweeps/          # current runs.tsv + resources/
    rsync -a ~/workspaces/rl_policy/dr_tools/monitor/data/ monitor/data/   # ~2.6 GB mirror (optional; else re-mirrored)
    MONITOR_DRY_RUN=1 monitor/tick.sh && tail monitor/logs/tick.log   # check, then install the cron line from section 3
