#!/usr/bin/env bash
# Follow-ups to the DR knob sweep:
#  1. Rerun knobs whose first pass used a flawed measurement (socket reference, grasp timing)
#     or ran before the observation-bias fix.
#  2. Joint-friction scan: at which friction does the arm stop responding? Headless, two push
#     strengths (0.3 = the sweep's excitation, 1.0 = full policy authority).
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../env.sh"
WT=$ISAACLAB_DIR
S="$WT/source"
OUT=$DR_TOOLS_RESULTS/knob_sweep

"$DR_TOOLS_HOME/dev/dr_knob_sweep.sh" socket_pos socket_rot grasp_pos grasp_rot obs_socket_pos obs_eef_pos

cd "$WT" || exit 1
export PYTHONPATH="$S/isaaclab:$S/isaaclab_tasks:$S/isaaclab_assets:$S/isaaclab_rl:$S/isaaclab_physx:${PYTHONPATH:-}"
unset DISPLAY
: > "$OUT/friction_scan.jsonl"
for PUSH in 0.3 1.0; do
  echo "=== friction scan push=$PUSH $(date +%H:%M:%S)" >> "$OUT/progress.log"
  SWEEP_KNOB=joint_friction SWEEP_FINAL=2.0,2.0 SWEEP_LEVELS=0,3,6,12,25,50 SWEEP_VIDEO=0 SWEEP_PUSH=$PUSH \
    $CONDA_RUN python "$DR_TOOLS_HOME/dev/dr_knob_sweep.py" < /dev/null \
    > "$OUT/friction_scan_push$PUSH.log" 2>&1
  grep "^SWEEP_RESULT" "$OUT/friction_scan_push$PUSH.log" | sed "s/^SWEEP_RESULT /{\"push\": $PUSH, \"r\": /; s/\$/}/" \
    >> "$OUT/friction_scan.jsonl" && echo "ok friction scan push=$PUSH" >> "$OUT/progress.log"
done
echo "FOLLOWUPS DONE $(date +%H:%M:%S)" >> "$OUT/progress.log"
