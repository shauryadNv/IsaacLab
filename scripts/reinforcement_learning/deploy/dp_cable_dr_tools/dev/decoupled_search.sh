#!/usr/bin/env bash
# Decoupled-OSC gain search for stable socket contact (payload compensation ON).
# Phase 1: insertion only per config -> results_insert.jsonl. Stable configs then get the full test set.
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../env.sh"; S=$ISAACLAB_DIR/source; OUT=$DR_TOOLS_RESULTS/decoupled_search
mkdir -p "$OUT"; cd "$ISAACLAB_DIR" || exit 1
export PYTHONPATH="$S/isaaclab:$S/isaaclab_tasks:$S/isaaclab_assets:$S/isaaclab_rl:$S/isaaclab_physx"; unset DISPLAY
run() { # label tests stiff damp partial nullspace -> appends to $OUT/results_$2.jsonl
  local label=$1 tests=$2
  LABEL=$label TESTS=$tests STIFF=$3 DAMP=$4 PARTIAL=$5 NULLSPACE=$6 INSERT_CLAMP=${INSERT_CLAMP:-0.3} DECOUPLE=1 COMP=1 \
    $CONDA_RUN python "$DR_TOOLS_HOME/dev/payload_comp_check.py" < /dev/null > "$OUT/$label.$tests.log" 2>&1
  python3 - "$OUT/$label.$tests.log" "$label" >> "$OUT/results_${tests//,/_}.jsonl" <<'PY'
import json, sys
row = {"label": sys.argv[2]}
for line in open(sys.argv[1], errors="replace"):
    if line.startswith("CHECK {"):
        d = json.loads(line[6:]); d.pop("diag", None); row[d.pop("test")] = d
print(json.dumps(row))
PY
}
[ "${PHASE2:-0}" = 1 ] || : > "$OUT/results_insert.jsonl"
# label | stiffness | damping | partial | nullspace
CONFIGS=(
  "k50-10_z1|50,50,50,10,10,10|1.0|0|"
  "k100-20_z1|100,100,100,20,20,20|1.0|0|"
  "k200-30_z1|200,200,200,30,30,30|1.0|0|"
  "k300-30_z1|300,300,300,30,30,30|1.0|0|"
  "k300-75_z1|300,300,300,75,75,75|1.0|0|"
  "k500-75_z1|500,500,500,75,75,75|1.0|0|"
  "k100-20_z0.7|100,100,100,20,20,20|0.7|0|"
  "k300-30_z1_partial|300,300,300,30,30,30|1.0|1|"
  "k100-20_z1_partial|100,100,100,20,20,20|1.0|1|"
  "k300-30_z1_ns|300,300,300,30,30,30|1.0|0|10:1.0:default"
  "k100-20_z1_ns|100,100,100,20,20,20|1.0|0|10:1.0:default"
  "k2000-300_z1_ns|2000,2000,2000,300,300,300|1.0|0|10:1.0:default"
)
[ "${PHASE2:-0}" = 1 ] || for c in "${CONFIGS[@]}"; do IFS='|' read -r label stiff damp partial ns <<< "$c"; echo "$(date +%T) $label" >> "$OUT/progress.log"; run "$label" insert "$stiff" "$damp" "$partial" "$ns"; done
echo "$(date +%T) PHASE1 DONE" >> "$OUT/progress.log"
[ "${PHASE2:-0}" = 1 ] || exit 0

# Phase 2: full tests + hard insertion (full-scale actions) for the contact-stable candidates.
: > "$OUT/results_hold_push_rot_insert.jsonl"; : > "$OUT/results_insert_hard.jsonl"
CANDIDATES=(
  "k100-20_z0.7|100,100,100,20,20,20|0.7"
  "k100-20_z1|100,100,100,20,20,20|1.0"
  "k200-30_z0.7|200,200,200,30,30,30|0.7"
  "k200-30_z1|200,200,200,30,30,30|1.0"
)
for c in "${CANDIDATES[@]}"; do
  IFS='|' read -r label stiff damp <<< "$c"; echo "$(date +%T) phase2 $label" >> "$OUT/progress.log"
  run "$label" hold,push,rot,insert "$stiff" "$damp" 0 ""
  INSERT_CLAMP=1.0 run "$label" insert "$stiff" "$damp" 0 "" && \
    tail -1 "$OUT/results_insert.jsonl" >> "$OUT/results_insert_hard.jsonl" && sed -i '$ d' "$OUT/results_insert.jsonl"
done
echo "$(date +%T) PHASE2 DONE" >> "$OUT/progress.log"
