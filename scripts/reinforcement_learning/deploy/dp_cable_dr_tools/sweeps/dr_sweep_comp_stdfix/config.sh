# Std-instability A/B: dr_sweep_comp variants that blew up, with ONE learned std shared across states
# (agent.policy.state_dependent_std=false). Same commit and config as dr_sweep_comp otherwise (paired controls).
DR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DR_DIR/../../env.sh"
WORKFLOW=$OSMO_WORKFLOW
OSMO_DIR=$(dirname "$WORKFLOW")
# COMMIT=b221791bbbfebed76d2b3106f753ec0b73920e7e   # shauryad/dp_cable_dr: + payload gravity compensation   (pre resume-LR fix)
COMMIT=794f5d5909b5d4914b8b21823eaf40bcf567c98b   # shauryad/dp_cable_dr: + resume keeps the checkpoint learning rate (relaunches)
POOL=isaac-dev-l40-04
# PRIORITY=LOW
PRIORITY=${PRIORITY:-LOW}   # env / resources/<name>.sh can raise it (monitor: last relaunch at HIGH)
IMAGE=nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1
ROBOT_TYPE_TASK=Rizon4s-Grav-TaskSpace
ROBOT_TYPE=rizon4s
# NUM_GPUS=1
NUM_GPUS=2
# NUM_ENVS=1024   # first launch (runs 1121-1142, cancelled)
NUM_ENVS=2048    # per GPU
# NUM_CPUS=15
NUM_CPUS=30
# MEMORY=48Gi
MEMORY=96Gi
# MAX_ITERATIONS=1500
MAX_ITERATIONS=6000
TAG_SUFFIX=-comp-stdfix-g2e2048
MANIFEST=$DR_DIR/manifest.tsv      # name <TAB> extra_overrides
RUNS=$DR_DIR/runs.tsv              # append-only: name <TAB> run_id <TAB> submitted_utc <TAB> reason
SWIFT_ROOT=$SWIFT_ROOT_BASE/displayport_insertion_$ROBOT_TYPE
