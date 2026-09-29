#!/usr/bin/env bash
# Run the live DR read-back at the easy and hard ends of the curriculum.
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../env.sh"
WT=$ISAACLAB_DIR
cd "$WT" || exit 1
S="$WT/source"
export PYTHONPATH="$S/isaaclab:$S/isaaclab_tasks:$S/isaaclab_assets:$S/isaaclab_rl:$S/isaaclab_physx:${PYTHONPATH:-}"
unset DISPLAY
for LEVEL in 0 50; do
  echo "===== LEVEL $LEVEL ====="
  $CONDA_RUN python $DR_TOOLS_HOME/dev/dr_verify_live.py "$LEVEL" 2>&1 \
    | grep -E "DR_LIVE_RESULT|Traceback|Error|error" | grep -viE "vulkan|swapchain|present"
done
