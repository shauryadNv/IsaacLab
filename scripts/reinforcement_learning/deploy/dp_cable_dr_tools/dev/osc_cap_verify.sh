#!/usr/bin/env bash
# Verify the capped OSC gain DR (stiffness <= 1.5x, damping ratio >= 0.7x) removes the insertion ejections.
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../env.sh"; DEV=$DR_TOOLS_HOME/dev; OUT=$DR_TOOLS_RESULTS/osc_cap_verify
mkdir -p "$OUT"; : > "$OUT/progress.log"
# V1/V2: knob sweep (new defaults) for the two knobs, insertion excitation
for combo in "comp1_clamp0.3|1|0.3" "comp0_clamp0.3|0|0.3" "comp1_clamp1.0|1|1.0"; do
  IFS='|' read -r tag comp push <<< "$combo"
  echo "$(date +%T) knob sweep $tag" >> "$OUT/progress.log"
  SWEEP_OUT=$OUT/knobs_$tag SWEEP_COMP=$comp SWEEP_EXCITE=insert SWEEP_PUSH=$push SWEEP_VIDEO=${VIDEO:-1} \
    /bin/bash $DEV/dr_knob_sweep.sh osc_stiffness osc_damping_ratio > "$OUT/knobs_$tag.driver.log" 2>&1
done
# V3: worst-case corner, both caps at once, 32 envs
S=$ISAACLAB_DIR/source; cd $ISAACLAB_DIR
export PYTHONPATH="$S/isaaclab:$S/isaaclab_tasks:$S/isaaclab_assets:$S/isaaclab_rl:$S/isaaclab_physx"; unset DISPLAY
: > "$OUT/corner.jsonl"
ZT=$(python3 -c "import math; print(0.7*35/(2*math.sqrt(300)))"); ZR=$(python3 -c "import math; print(0.7*1.1/(2*math.sqrt(30)))")
for comp in 1 0; do for cl in 0.3 1.0; do
  label=corner_k1.5_z0.7_comp${comp}_clamp$cl; echo "$(date +%T) $label" >> "$OUT/progress.log"
  LABEL=$label N_ENVS=32 TESTS=insert INSERT_CLAMP=$cl STIFF=450,450,450,45,45,45 DAMP=$ZT,$ZT,$ZT,$ZR,$ZR,$ZR COMP=$comp \
    $CONDA_RUN python $DEV/payload_comp_check.py < /dev/null > "$OUT/$label.log" 2>&1
  python3 - "$OUT/$label.log" "$label" >> "$OUT/corner.jsonl" <<'PY'
import json, sys
row = {"label": sys.argv[2]}
for line in open(sys.argv[1], errors="replace"):
    if line.startswith("CHECK {"):
        d = json.loads(line[6:]); d.pop("diag", None); row[d.pop("test")] = d
print(json.dumps(row))
PY
done; done
echo "$(date +%T) DONE" >> "$OUT/progress.log"
