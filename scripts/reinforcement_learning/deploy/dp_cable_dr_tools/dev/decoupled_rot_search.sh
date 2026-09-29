#!/usr/bin/env bash
# Decoupled OSC: soft translation (K 100) with stiff rotation, and the same gains without payload compensation.
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../env.sh"; S=$ISAACLAB_DIR/source; OUT=$DR_TOOLS_RESULTS/decoupled_search
cd "$ISAACLAB_DIR" || exit 1
export PYTHONPATH="$S/isaaclab:$S/isaaclab_tasks:$S/isaaclab_assets:$S/isaaclab_rl:$S/isaaclab_physx"; unset DISPLAY
: > "$OUT/results_rot.jsonl"
one() { # label tests clamp stiff damp comp
  LABEL=$1 TESTS=$2 INSERT_CLAMP=$3 STIFF=$4 DAMP=$5 COMP=$6 DECOUPLE=1 \
    $CONDA_RUN python "$DR_TOOLS_HOME/dev/payload_comp_check.py" < /dev/null > "$OUT/$1.$2.$3.log" 2>&1
  python3 - "$OUT/$1.$2.$3.log" "$1" "$3" >> "$OUT/results_rot.jsonl" <<'PY'
import json, sys
row = {"label": sys.argv[2], "clamp": float(sys.argv[3])}
for line in open(sys.argv[1], errors="replace"):
    if line.startswith("CHECK {"):
        d = json.loads(line[6:]); d.pop("diag", None); row[d.pop("test")] = d
print(json.dumps(row))
PY
}
for c in "k100-300_z1|100,100,100,300,300,300" "k100-1000_z1|100,100,100,1000,1000,1000" "k100-3000_z1|100,100,100,3000,3000,3000"; do
  IFS='|' read -r label stiff <<< "$c"; echo "$(date +%T) rot $label" >> "$OUT/progress.log"
  one "$label" hold,push,rot,insert 0.3 "$stiff" 1.0 1
  one "$label" insert 1.0 "$stiff" 1.0 1
done
# Same decoupled gains WITHOUT payload compensation (does decoupling alone remove the sag?)
one "k100-75_z1_NOCOMP" hold,push,rot,insert 0.3 "100,100,100,75,75,75" 1.0 0
one "k100-1000_z1_NOCOMP" hold,push,rot 0.3 "100,100,100,1000,1000,1000" 1.0 0
echo "$(date +%T) ROT DONE" >> "$OUT/progress.log"
