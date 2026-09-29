# 2 GPU x 2048 envs hit CUDA OOM on GPU 1 at iteration 1 in dr_sweep (run 1171); kept at 4 GPU x 1024
# so this run stays comparable with its siblings in the other sweeps.
NUM_GPUS=4
NUM_ENVS=1024
NUM_CPUS=40
MEMORY=128Gi
TAG_SUFFIX=-noatgoal-comp-g4e1024
