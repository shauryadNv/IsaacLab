#!/usr/bin/env bash
# Which OSC gain axes make contact unstable, and which caps are safe? Worst-case corner tests
# (every env at the extreme), payload compensation ON, full-scale scripted insertion, 32 envs.
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../env.sh"; O=$DR_TOOLS_RESULTS/osc_cap_verify; S=$ISAACLAB_DIR/source
cd $ISAACLAB_DIR; export PYTHONPATH="$S/isaaclab:$S/isaaclab_tasks:$S/isaaclab_assets:$S/isaaclab_rl:$S/isaaclab_physx"; unset DISPLAY
OUTF=${OUTF:-diag}; : > "$O/$OUTF.jsonl"
zt() { python3 -c "import math; print($1*35/(2*math.sqrt(300)))"; }; zr() { python3 -c "import math; print($1*1.1/(2*math.sqrt(30)))"; }
# label | stiffness (x,y,z,rx,ry,rz) | damping ratio (x,y,z,rx,ry,rz)
CONFIGS=(
  "nominal|300,300,300,30,30,30|$(zt 1),$(zt 1),$(zt 1),$(zr 1),$(zr 1),$(zr 1)"
  "trans_only_k1.5_z0.7|450,450,450,30,30,30|$(zt 0.7),$(zt 0.7),$(zt 0.7),$(zr 1),$(zr 1),$(zr 1)"
  "rot_only_k1.5_z0.7|300,300,300,45,45,45|$(zt 1),$(zt 1),$(zt 1),$(zr 0.7),$(zr 0.7),$(zr 0.7)"
  "both_k1.25_z0.85|375,375,375,37.5,37.5,37.5|$(zt 0.85),$(zt 0.85),$(zt 0.85),$(zr 0.85),$(zr 0.85),$(zr 0.85)"
  "both_k1.2_z0.9|360,360,360,36,36,36|$(zt 0.9),$(zt 0.9),$(zt 0.9),$(zr 0.9),$(zr 0.9),$(zr 0.9)"
  "trans_only_k1.5_z0.7_rot_z1.5|450,450,450,30,30,30|$(zt 0.7),$(zt 0.7),$(zt 0.7),$(zr 1.5),$(zr 1.5),$(zr 1.5)"
)
[ -n "${CONFIG_FILE:-}" ] && mapfile -t CONFIGS < <(bash -c "zt() { python3 -c \"import math; print(\$1*35/(2*math.sqrt(300)))\"; }; zr() { python3 -c \"import math; print(\$1*1.1/(2*math.sqrt(30)))\"; }; while read -r l; do eval echo \"\$l\"; done < $CONFIG_FILE")
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r label stiff damp <<< "$c"
  LABEL=$label N_ENVS=32 TESTS=insert INSERT_CLAMP=1.0 STIFF=$stiff DAMP=$damp COMP=${COMPV:-1} \
    $CONDA_RUN python $DR_TOOLS_HOME/dev/payload_comp_check.py < /dev/null > "$O/${OUTF}_$label.log" 2>&1
  python3 - "$O/${OUTF}_$label.log" "$label" >> "$O/$OUTF.jsonl" <<'PY'
import json, sys
row = {"label": sys.argv[2]}
for line in open(sys.argv[1], errors="replace"):
    if line.startswith("CHECK {"):
        d = json.loads(line[6:]); d.pop("diag", None); row[d.pop("test")] = d
print(json.dumps(row))
PY
done
python3 - <<'PY'
import json
import os
for l in open(os.environ['DR_TOOLS_RESULTS']+'/osc_cap_verify/'+os.environ.get('OUTF','diag')+'.jsonl'):
    r=json.loads(l); i=r.get('insert',{})
    print(f"{r['label']:32s} succ {i.get('success_frac')} seated {i.get('seated_steps_mean')} dropped {i.get('dropped')} blowup {i.get('blowup_envs')} peak_tau_med {i.get('peak_tau_median')}")
PY
