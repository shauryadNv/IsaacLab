
# 2026-09-28T11:20:17Z monitor: CUDA OOM on isaaclab_train_rsl_rl-1297 -> fall back to 4 GPU x 1024 envs (same total envs).
NUM_GPUS=4
NUM_ENVS=1024
NUM_CPUS=40
MEMORY=128Gi
TAG_SUFFIX=${TAG_SUFFIX/g2e2048/g4e1024}
