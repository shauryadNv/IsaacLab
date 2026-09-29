#!/usr/bin/env bash
# Submit one DR-sweep run by manifest name and record its osmo run id.
#   ./submit.sh <name> [reason]              fresh run, e.g. ./submit.sh all_on initial
#   ./submit.sh --dry-run <name>             print the rendered workflow, submit nothing
# Resume a dead run (used by the monitor):
#   ATTEMPT=2 RESUME_FROM=<swift run folder> MAX_ITERS=<remaining> ./submit.sh <name> relaunch
# Each attempt uploads to its own folder (-dr_<name>, -dr_<name>-r2, ...) so a resumed run
# never overwrites the checkpoints it resumed from.
set -u
source "$(dirname "$0")/config.sh"

DRY=""
if [ "${1:-}" = "--dry-run" ]; then DRY="--dry-run"; shift; fi
NAME=${1:?usage: submit.sh [--dry-run] <name> [reason]}
REASON=${2:-initial}
ATTEMPT=${ATTEMPT:-1}
RESUME_FROM=${RESUME_FROM:-}
MAX_ITERS=${MAX_ITERS:-$MAX_ITERATIONS}

# Per-run resource overrides (e.g. a run that OOMs at the sweep default): resources/<name>.sh
[ -f "$DR_DIR/resources/$NAME.sh" ] && source "$DR_DIR/resources/$NAME.sh"

EXTRA=$(awk -F'\t' -v n="$NAME" 'NR > 1 && $1 == n { print $2 }' "$MANIFEST")
if [ -z "$EXTRA" ]; then echo "unknown run name: $NAME" >&2; exit 2; fi
# RUN_TAG="-dr_$NAME"
RUN_TAG="-dr_$NAME${TAG_SUFFIX:-}"
[ "$ATTEMPT" -gt 1 ] && RUN_TAG="$RUN_TAG-r$ATTEMPT"

OUT=$(cd "$OSMO_DIR" && osmo workflow submit "$WORKFLOW" -p "$POOL" --priority "$PRIORITY" $DRY --set \
  image="$IMAGE" commit_hash="$COMMIT" robot_type_task="$ROBOT_TYPE_TASK" robot_type="$ROBOT_TYPE" \
  num_gpus="$NUM_GPUS" num_envs="$NUM_ENVS" num_cpus="$NUM_CPUS" memory="$MEMORY" \
  max_iterations="$MAX_ITERS" train_script=scripts/reinforcement_learning/train.py \
  "rl_library_arg=--rl_library rsl_rl" task_prefix=IsaacContrib-Deploy-DisplayportInsertion- \
  task_version= run_tag="$RUN_TAG" resume_from="$RESUME_FROM" "extra_overrides=$EXTRA" 2>&1)

if [ -n "$DRY" ]; then echo "$OUT"; exit 0; fi
RID=$(echo "$OUT" | grep -oE "isaaclab_train_rsl_rl-[0-9]+" | head -1)
if [ -z "$RID" ]; then
  echo "SUBMIT FAILED for $NAME: $(echo "$OUT" | grep -iE 'error|status code' | tail -2)" >&2
  exit 1
fi
[ -f "$RUNS" ] || printf "name\trun_id\tsubmitted_utc\treason\tattempt\trun_tag\tresume_from\tmax_iters\tcommit\tpriority\n" > "$RUNS"
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$NAME" "$RID" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$REASON" \
  "$ATTEMPT" "$RUN_TAG" "$RESUME_FROM" "$MAX_ITERS" "$COMMIT" "$PRIORITY" >> "$RUNS"
echo "$NAME -> $RID ($PRIORITY)"
