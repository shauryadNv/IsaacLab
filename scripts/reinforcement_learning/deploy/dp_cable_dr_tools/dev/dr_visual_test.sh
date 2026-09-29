#!/usr/bin/env bash
# Visual check of the DisplayPort domain randomization (branch shauryad/dp_cable_dr).
#
#   ./dr_visual_test.sh gui      live Isaac Sim window. Run from a terminal INSIDE your
#                                desktop session (DISPLAY=:1 etc.), not over ssh -X.
#   ./dr_visual_test.sh video    headless; records mp4s to
#                                IsaacLab_dr/logs/rsl_rl/displayport_insertion_rizon4s/<run>/videos/train/
#
# Extra Hydra overrides after the mode are passed through, e.g. exaggerate the grasp
# rotation so it is obvious by eye:
#   ./dr_visual_test.sh gui 'env.dr.grasp_rot.final=[0.3,0.3,0.3]'
#
# NUM_ENVS / ITERS env vars override the defaults below.
set -euo pipefail

MODE="${1:-gui}"
shift || true

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../env.sh"
WT=$ISAACLAB_DIR
S="$WT/source"
cd "$WT"
# Point the conda env's Python at this worktree instead of IsaacLab_ship.
export PYTHONPATH="$S/isaaclab:$S/isaaclab_tasks:$S/isaaclab_assets:$S/isaaclab_rl:$S/isaaclab_physx${PYTHONPATH:+:$PYTHONPATH}"

NUM_ENVS="${NUM_ENVS:-16}"
ITERS="${ITERS:-20}"

VIEW_ARGS=(--viz kit)
# Close-up on env 0's socket. Anchored to the env, not the socket, so socket-pose
# randomization is visible as the socket moving between resets. The socket opening sits
# at (0.475, 0.125, 0.06) env-local and faces +Z, so the plug arrives from above.
# The default camera sits metres away, where the plug is a few pixels.
CAM_ARGS=(
  env.viewer.origin_type=env
  env.viewer.env_index=0
  'env.viewer.eye=[0.75,0.35,0.25]'
  'env.viewer.lookat=[0.475,0.125,0.08]'
)
case "$MODE" in
  gui) ;;
  video)
    # Offscreen: without a display Kit renders the video without trying to open a window.
    unset DISPLAY
    VIEW_ARGS+=(--video --video_length 200 --video_interval 400)
    ;;
  *) echo "usage: $0 [gui|video] [extra hydra overrides...]" >&2; exit 2 ;;
esac

DR_ARGS=(
  env.dr.enabled=true
  # Start the curriculum at its top level so every range is at its widest from the first
  # reset. At level 0 most ranges are zero and the env would look unrandomized.
  env.dr.adr.enable=true
  env.dr.adr.init_level=50
  # At the top level this spawns every plug at the approach pose, so the grasp is visible.
  env.dr.at_goal_schedule=adr
  # Controller / plant (not visible directly; shows up as motion quality)
  env.dr.osc_stiffness.enable=true
  env.dr.osc_damping_ratio.enable=true
  env.dr.joint_armature.enable=true
  env.dr.joint_friction.enable=true
  # Contact
  env.dr.finger_friction.enable=true
  env.dr.plug_mass.enable=true
  # Reset state (the most visible ones)
  env.dr.grasp_pos.enable=true
  env.dr.grasp_rot.enable=true
  env.dr.socket_pos.enable=true
  env.dr.socket_rot.enable=true
  # Observation noise
  env.dr.obs_socket_pos.enable=true
  env.dr.obs_eef_pos.enable=true
  env.dr.obs_eef_rot.enable=true
  env.dr.obs_socket_rot.enable=true
  # Actions (visible as jitter / lag)
  env.dr.action_noise.enable=true
  env.dr.action_latency.enable=true
  # Cable-tug disturbance on the plug
  env.dr.plug_wrench_force.enable=true
)

exec $CONDA_RUN python scripts/reinforcement_learning/train.py \
  --rl_library rsl_rl \
  --task IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-ROS-Inference \
  --num_envs "$NUM_ENVS" \
  "${VIEW_ARGS[@]}" \
  agent.max_iterations="$ITERS" \
  agent.run_name=dr_visual \
  "${CAM_ARGS[@]}" \
  "${DR_ARGS[@]}" \
  "$@"
