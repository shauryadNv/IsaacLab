#!/usr/bin/env bash
# Runs the finger-friction slip test (4 conditions) and records the wrench demo videos
# (3 conditions), then tiles the videos side by side.
#   Results:  dr_tools/results/wrench_videos/results.txt
#   Grid:     dr_tools/results/wrench_videos/wrench_grid.mp4
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../env.sh"
WT=$ISAACLAB_DIR
S="$WT/source"
OUT=$DR_TOOLS_RESULTS/wrench_videos
mkdir -p "$OUT"
cd "$WT" || exit 1
export PYTHONPATH="$S/isaaclab:$S/isaaclab_tasks:$S/isaaclab_assets:$S/isaaclab_rl:$S/isaaclab_physx:${PYTHONPATH:-}"
unset DISPLAY
RES=$OUT/results.txt
: > "$RES"

run() {  # run <marker> <script>, keeping only the marker line (and any traceback)
  $CONDA_RUN python "$2" < /dev/null 2>&1 | tee -a "$OUT/raw.log" \
    | grep -E "$1|Traceback|Error:" | grep -viE "vulkan|swapchain|present" >> "$RES"
}

echo "##### SLIP TEST #####" >> "$RES"
for FMU in 0.4 1.1; do
  for PMU in 0.001 1.0; do
    export SLIP_FINGER_MU=$FMU SLIP_PLUG_MU=$PMU
    run SLIP_RESULT "$DR_TOOLS_HOME/dev/dr_friction_slip_test.py"
  done
done

echo "##### WRENCH DEMO #####" >> "$RES"
while read -r NAME F; do
  export DEMO_NAME=$NAME DEMO_FORCE=$F DEMO_OUT=$OUT
  run WRENCH_DEMO "$DR_TOOLS_HOME/dev/dr_wrench_demo.py"
done <<'EOF'
1_none 0
2_force_real 0.6
3_force_5x 3.0
EOF

# Tile the three clips side by side with labels.
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
IN=(); FILT=""; i=0
for NAME in 1_none 2_force_real 3_force_5x; do
  CLIP=$(ls "$OUT/videos/$NAME"/*.mp4 2>/dev/null | head -1)
  [ -z "$CLIP" ] && { echo "missing clip for $NAME" >> "$RES"; continue; }
  IN+=(-i "$CLIP")
  LABEL=${NAME#*_}
  if [ -f "$FONT" ]; then
    FILT+="[$i:v]scale=640:360,drawtext=fontfile=$FONT:text='$LABEL':x=12:y=12:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.55[v$i];"
  else
    FILT+="[$i:v]scale=640:360[v$i];"
  fi
  i=$((i+1))
done
if [ "$i" -eq 3 ]; then
  FILT+="[v0][v1][v2]hstack=3[out]"
  ffmpeg -loglevel error -y "${IN[@]}" -filter_complex "$FILT" -map "[out]" -shortest "$OUT/wrench_grid.mp4" \
    && echo "grid: $OUT/wrench_grid.mp4" >> "$RES"
fi
echo "DONE" >> "$RES"
