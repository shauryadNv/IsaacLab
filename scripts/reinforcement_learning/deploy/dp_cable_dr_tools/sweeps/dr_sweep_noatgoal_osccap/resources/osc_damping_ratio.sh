# Kept identical to its curriculum-on sibling so the pair stays comparable.
# 2 GPU x 2048 envs hit CUDA OOM on GPU 1 at iteration 1 (run 1171); user's fallback is 4 GPU x 1024.
NUM_GPUS=4
NUM_ENVS=1024
NUM_CPUS=40
MEMORY=128Gi
TAG_SUFFIX=-noatgoal-osccap-g4e1024
