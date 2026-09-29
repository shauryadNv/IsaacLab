#!/usr/bin/env bash
# Record payload_comp_check.py with compensation OFF and ON and build side-by-side videos.
#   Output: dr_tools/results/payload_comp/{hold,push,rot_yaw,insert}_off_vs_on.mp4 and all_tests_off_vs_on.mp4
# Green sphere = the plug's position at the start of each test (drift/sag is measured against it).
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../env.sh"
WT=$ISAACLAB_DIR
S=$WT/source
OUT=$DR_TOOLS_RESULTS/payload_comp
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
mkdir -p "$OUT"
cd "$WT" || exit 1
export PYTHONPATH="$S/isaaclab:$S/isaaclab_tasks:$S/isaaclab_assets:$S/isaaclab_rl:$S/isaaclab_physx:${PYTHONPATH:-}"
unset DISPLAY

for C in 0 1; do
  [ "${SKIP_SIM:-0}" = 1 ] && break
  rm -rf "$OUT/videos/comp$C"
  COMP=$C VIDEO=1 OUT=$OUT $CONDA_RUN python "$DR_TOOLS_HOME/dev/payload_comp_check.py" \
    < /dev/null > "$OUT/comp$C.log" 2>&1
  grep "^CHECK" "$OUT/comp$C.log" > "$OUT/comp$C.results.txt"
done

OFF=$(ls "$OUT"/videos/comp0/*.mp4 | head -1)
ON=$(ls "$OUT"/videos/comp1/*.mp4 | head -1)
[ -f "$OFF" ] && [ -f "$ON" ] || { echo "missing clips"; exit 1; }

TESTS=(hold push rot_yaw insert)
# (no colons: drawtext treats ":" as an option separator)
TITLES=("hold - zero action for 5 s" "push - +x then -x at 0.3" "rotate - +yaw then -yaw at 0.3" "insert - scripted descent into the socket")
PARTS=()
for i in 0 1 2 3; do
  st=$((i * 5))
  F="[0:v]trim=start=$st:duration=5,setpts=PTS-STARTPTS,scale=800:450,"
  F+="drawtext=fontfile=$FONT:text='compensation OFF':x=10:y=10:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.6,"
  F+="drawtext=fontfile=$FONT:text='${TITLES[$i]}':x=10:y=h-40:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6[a];"
  F+="[1:v]trim=start=$st:duration=5,setpts=PTS-STARTPTS,scale=800:450,"
  F+="drawtext=fontfile=$FONT:text='compensation ON':x=10:y=10:fontsize=26:fontcolor=white:box=1:boxcolor=0x1c5cab@0.85[b];"
  F+="[a][b]hstack=2[out]"
  ffmpeg -loglevel error -y -i "$OFF" -i "$ON" -filter_complex "$F" -map "[out]" "$OUT/${TESTS[$i]}_off_vs_on.mp4"
  PARTS+=("$OUT/${TESTS[$i]}_off_vs_on.mp4")
done
printf "file '%s'\n" "${PARTS[@]}" > "$OUT/concat.txt"
ffmpeg -loglevel error -y -f concat -safe 0 -i "$OUT/concat.txt" -c copy "$OUT/all_tests_off_vs_on.mp4"
ls -la "$OUT"/*.mp4
