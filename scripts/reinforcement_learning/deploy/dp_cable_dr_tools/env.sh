# Machine-specific settings for the DP cable DR tools. Sourced by every script here.
# Each value can be overridden by exporting it before running a script.

# This directory (dp_cable_dr_tools) and the Isaac Lab checkout it lives in.
DR_TOOLS_HOME="${DR_TOOLS_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ISAACLAB_DIR="${ISAACLAB_DIR:-$(cd "$DR_TOOLS_HOME/../../../.." && pwd)}"

# Conda env with Isaac Sim + this Isaac Lab (used for local sims: rollouts, knob sweeps, checks).
# `conda run` applies the env's activation hooks, which Isaac Sim needs.
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-isaaclab_ship}"
ISAACLAB_PYTHON="${ISAACLAB_PYTHON:-$(dirname "$CONDA_BIN")/../envs/$CONDA_ENV/bin/python}"
CONDA_RUN="${CONDA_RUN:-$CONDA_BIN run --no-capture-output -n $CONDA_ENV}"
# Isaac Lab source packages from this checkout take precedence over any installed copies.
export PYTHONPATH="$ISAACLAB_DIR/source/isaaclab:$ISAACLAB_DIR/source/isaaclab_tasks:$ISAACLAB_DIR/source/isaaclab_assets:$ISAACLAB_DIR/source/isaaclab_rl:$ISAACLAB_DIR/source/isaaclab_physx${PYTHONPATH:+:$PYTHONPATH}"

# osmo training: workflow template and where runs upload checkpoints / TensorBoard logs.
OSMO_WORKFLOW="${OSMO_WORKFLOW:-$DR_TOOLS_HOME/osmo/dp_cable.yaml}"
SWIFT_ROOT_BASE="${SWIFT_ROOT_BASE:-swift://pdx.s8k.io/AUTH_team-isaac/datasets/shauryad}"

# Where local test scripts write their outputs (videos, JSON results).
DR_TOOLS_RESULTS="${DR_TOOLS_RESULTS:-$DR_TOOLS_HOME/results}"

export DR_TOOLS_HOME ISAACLAB_DIR CONDA_BIN CONDA_ENV ISAACLAB_PYTHON CONDA_RUN OSMO_WORKFLOW SWIFT_ROOT_BASE DR_TOOLS_RESULTS
