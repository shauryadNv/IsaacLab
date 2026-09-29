#!/usr/bin/env bash
# Runs dr_knob_sweep.py for each knob given (default: all 19) and builds, per knob, a
# side-by-side video "level 0 | level 25 | level 50".
#   Results: dr_tools/results/knob_sweep/results.jsonl     Videos: dr_tools/results/knob_sweep/<knob>_levels.mp4
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../env.sh"
WT=$ISAACLAB_DIR
S="$WT/source"
OUT=${SWEEP_OUT:-$DR_TOOLS_RESULTS/knob_sweep}
PHASE=${SWEEP_PHASE_STEPS:-150}
mkdir -p "$OUT"
cd "$WT" || exit 1
export PYTHONPATH="$S/isaaclab:$S/isaaclab_tasks:$S/isaaclab_assets:$S/isaaclab_rl:$S/isaaclab_physx:${PYTHONPATH:-}"
export SWEEP_OUT=$OUT SWEEP_PHASE_STEPS=$PHASE
unset DISPLAY
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf

KNOBS=("$@")
if [ ${#KNOBS[@]} -eq 0 ]; then
  KNOBS=(osc_stiffness osc_damping_ratio joint_armature joint_friction finger_friction mating_friction
         plug_mass grasp_pos grasp_rot socket_pos socket_rot obs_socket_pos obs_eef_pos obs_eef_rot
         obs_socket_rot action_noise action_latency plug_wrench_force)
fi

for KNOB in "${KNOBS[@]}"; do
  export SWEEP_KNOB=$KNOB
  echo "=== $KNOB  $(date +%H:%M:%S)" >> "$OUT/progress.log"
  $CONDA_RUN python "$DR_TOOLS_HOME/dev/dr_knob_sweep.py" < /dev/null > "$OUT/$KNOB.log" 2>&1
  if grep -q "^SWEEP_RESULT" "$OUT/$KNOB.log"; then
    grep "^SWEEP_RESULT" "$OUT/$KNOB.log" | sed 's/^SWEEP_RESULT //' >> "$OUT/results.jsonl"
    echo "ok $KNOB" >> "$OUT/progress.log"
  else
    echo "FAILED $KNOB: $(grep -E 'Error|Traceback' "$OUT/$KNOB.log" | tail -2 | tr '\n' ' ')" >> "$OUT/progress.log"
    continue
  fi
  CLIP=$(ls "$OUT/videos/$KNOB"/*.mp4 2>/dev/null | head -1)
  [ -z "$CLIP" ] && { echo "no clip for $KNOB" >> "$OUT/progress.log"; continue; }
  SEG=$(python3 -c "print($PHASE/30)")
  F=""
  for i in 0 1 2; do
    L=$((i * 25))
    F+="[0:v]trim=start=$(python3 -c "print($i*$SEG)"):duration=$SEG,setpts=PTS-STARTPTS,scale=640:360,"
    F+="drawtext=fontfile=$FONT:text='$KNOB  level $L':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.55[v$i];"
  done
  F+="[v0][v1][v2]hstack=3[out]"
  ffmpeg -loglevel error -y -i "$CLIP" -filter_complex "$F" -map "[out]" "$OUT/${KNOB}_levels.mp4" \
    && echo "video $OUT/${KNOB}_levels.mp4" >> "$OUT/progress.log"
done
echo "SWEEP DONE $(date +%H:%M:%S)" >> "$OUT/progress.log"
