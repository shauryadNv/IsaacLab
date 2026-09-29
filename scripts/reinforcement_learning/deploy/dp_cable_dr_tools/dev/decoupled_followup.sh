#!/usr/bin/env bash
# Follow-up: raise rotational stiffness at K_trans=100 (decoupled, comp ON) and stress the DR extremes.
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../env.sh"; S=$ISAACLAB_DIR/source; OUT=$DR_TOOLS_RESULTS/decoupled_search
cd "$ISAACLAB_DIR" || exit 1
export PYTHONPATH="$S/isaaclab:$S/isaaclab_tasks:$S/isaaclab_assets:$S/isaaclab_rl:$S/isaaclab_physx"; unset DISPLAY
: > "$OUT/results_followup.jsonl"
one() { # label tests clamp stiff damp
  LABEL=$1 TESTS=$2 INSERT_CLAMP=$3 STIFF=$4 DAMP=$5 DECOUPLE=1 COMP=1 \
    $CONDA_RUN python "$DR_TOOLS_HOME/dev/payload_comp_check.py" < /dev/null > "$OUT/$1.$2.$3.log" 2>&1
  python3 - "$OUT/$1.$2.$3.log" "$1" "$3" >> "$OUT/results_followup.jsonl" <<'PY'
import json, sys
row = {"label": sys.argv[2], "clamp": float(sys.argv[3])}
for line in open(sys.argv[1], errors="replace"):
    if line.startswith("CHECK {"):
        d = json.loads(line[6:]); d.pop("diag", None); row[d.pop("test")] = d
print(json.dumps(row))
PY
}
for c in "k100-50_z1|100,100,100,50,50,50|1.0" "k100-75_z1|100,100,100,75,75,75|1.0" "k100-100_z1|100,100,100,100,100,100|1.0"; do
  IFS='|' read -r label stiff damp <<< "$c"; echo "$(date +%T) followup $label" >> "$OUT/progress.log"
  one "$label" hold,push,rot,insert 0.3 "$stiff" "$damp"
  one "$label" insert 1.0 "$stiff" "$damp"
done
# DR-extreme stress for the K100 family: stiffness x2, damping ratio x0.5 (osc_stiffness / osc_damping_ratio at ADR 50)
one "k200-100_z0.5_DRextreme" insert 1.0 "200,200,200,100,100,100" "0.5"
one "k200-150_z0.5_DRextreme" insert 1.0 "200,200,200,150,150,150" "0.5"
echo "$(date +%T) FOLLOWUP DONE" >> "$OUT/progress.log"
