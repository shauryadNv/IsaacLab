# Main-line ADR follow-up: no near-goal resets, payload compensation, global
# policy standard deviation, and contact-stable OSC gain ranges.
DR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DR_DIR/../../env.sh"
WORKFLOW=$OSMO_WORKFLOW
OSMO_DIR=$(dirname "$WORKFLOW")
COMMIT=ddda77447e
POOL=isaac-dev-l40-04
PRIORITY=${PRIORITY:-LOW}
IMAGE=nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1
ROBOT_TYPE_TASK=Rizon4s-Grav-TaskSpace
ROBOT_TYPE=rizon4s
NUM_GPUS=2
NUM_ENVS=2048
NUM_CPUS=30
MEMORY=96Gi
MAX_ITERATIONS=6000
STEPS_PER_ITERATION=512
TAG_SUFFIX=-noatgoal-comp-osccap-adrfix-g2e2048
MANIFEST=$DR_DIR/manifest.tsv
RUNS=$DR_DIR/runs.tsv
SWIFT_ROOT=$SWIFT_ROOT_BASE/displayport_insertion_$ROBOT_TYPE
