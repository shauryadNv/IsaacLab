#!/usr/bin/env bash
# Softer / more-damped-only OSC gain corners (32 envs, full-scale scripted insertion), comp ON and OFF.
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../env.sh"; O=$DR_TOOLS_RESULTS/osc_cap_verify; S=$ISAACLAB_DIR/source
cd $ISAACLAB_DIR; export PYTHONPATH="$S/isaaclab:$S/isaaclab_tasks:$S/isaaclab_assets:$S/isaaclab_rl:$S/isaaclab_physx"; unset DISPLAY
for comp in 1 0; do
  out=$O/diag2_comp$comp.jsonl; : > $out
  while IFS='|' read -r label stiff damp; do
    [ -z "$label" ] && continue
    LABEL=$label N_ENVS=32 TESTS=insert INSERT_CLAMP=1.0 STIFF=$stiff DAMP=$damp COMP=$comp \
      $CONDA_RUN python $DR_TOOLS_HOME/dev/payload_comp_check.py < /dev/null > "$O/diag2_comp${comp}_$label.log" 2>&1
    python3 - "$O/diag2_comp${comp}_$label.log" "$label" >> $out <<'PY'
import json, sys
row = {"label": sys.argv[2]}
for line in open(sys.argv[1], errors="replace"):
    if line.startswith("CHECK {"):
        d = json.loads(line[6:]); d.pop("diag", None); row[d.pop("test")] = d
print(json.dumps(row))
PY
  done < $DR_TOOLS_HOME/dev/osc_cap_diag2.configs
done
for comp in 1 0; do echo "== comp $comp"; python3 -c "
import json
for l in open('$O/diag2_comp$comp.jsonl'):
    r=json.loads(l); i=r.get('insert',{})
    print(f\"{r['label']:32s} succ {i.get('success_frac')} seated {i.get('seated_steps_mean')} first {i.get('first_success_step_mean')} dropped {i.get('dropped')} blowup {i.get('blowup_envs')}\")"; done
