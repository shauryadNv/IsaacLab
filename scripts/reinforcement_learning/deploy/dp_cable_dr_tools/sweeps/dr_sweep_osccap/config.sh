# Capped OSC gain DR twin of dr_sweep: osc_stiffness final (0.5, 1.0), osc_damping_ratio final (1.0, 1.5)
# (softer / more damped only; stiffer or less damped translation makes insertion contact blow up).
# Variants that use those knobs: osc_stiffness, osc_damping_ratio, grp_controller, all_on.
DR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DR_DIR/../../env.sh"
WORKFLOW=$OSMO_WORKFLOW
OSMO_DIR=$(dirname "$WORKFLOW")
COMMIT=c93192d3619bd34a0f1b591f74fbed1a5a9007b5   # shauryad/dp_cable_dr: + softer/more-damped-only OSC gain DR
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
TAG_SUFFIX=-osccap-g2e2048
MANIFEST=$DR_DIR/manifest.tsv      # name <TAB> extra_overrides
RUNS=$DR_DIR/runs.tsv              # append-only: name <TAB> run_id <TAB> submitted_utc <TAB> reason
SWIFT_ROOT=$SWIFT_ROOT_BASE/displayport_insertion_$ROBOT_TYPE
