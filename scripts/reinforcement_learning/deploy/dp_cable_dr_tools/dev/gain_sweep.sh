#!/usr/bin/env bash
# Can controller gains / joint friction alone remove the plug-weight sag? Runs payload_comp_check.py
# (compensation OFF) per config and collects results into dr_tools/results/gain_sweep/results.jsonl.
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../env.sh"
S=$ISAACLAB_DIR/source
OUT=$DR_TOOLS_RESULTS/gain_sweep
mkdir -p "$OUT"; : > "$OUT/results.jsonl"
cd "$ISAACLAB_DIR" || exit 1
export PYTHONPATH="$S/isaaclab:$S/isaaclab_tasks:$S/isaaclab_assets:$S/isaaclab_rl:$S/isaaclab_physx:${PYTHONPATH:-}"
unset DISPLAY
# label | stiffness | damping ratio | joint friction
CONFIGS=(
  "base_300-30|||"
  "k300-30_z0.7|300,300,300,30,30,30|0.7|"
  "k2000-300_z0.7|2000,2000,2000,300,300,300|0.7|"
  "k500-75_z0.7|500,500,500,75,75,75|0.7|"
  "base_fric0.05|||0.05"
  "base_fric0.1|||0.1"
  "base_fric0.2|||0.2"
  "k500-75_z0.7_fric0.1|500,500,500,75,75,75|0.7|0.1"
  "k2000-300_z0.7_fric0.1|2000,2000,2000,300,300,300|0.7|0.1"
)
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r label stiff damp fric <<< "$c"
  echo "$(date +%T) $label" >> "$OUT/progress.log"
  LABEL=$label STIFF=$stiff DAMP=$damp JFRIC=$fric COMP=0 \
    $CONDA_RUN python "$DR_TOOLS_HOME/dev/payload_comp_check.py" \
    < /dev/null > "$OUT/$label.log" 2>&1
  python3 - "$OUT/$label.log" "$label" >> "$OUT/results.jsonl" <<'PY'
import json, sys
row = {"label": sys.argv[2]}
for line in open(sys.argv[1], errors="replace"):
    if line.startswith("CHECK {"):
        d = json.loads(line[6:]); row[d.pop("test")] = d
print(json.dumps(row))
PY
done
echo "$(date +%T) DONE" >> "$OUT/progress.log"
