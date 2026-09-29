
# 2026-09-27T19:06:01Z monitor: CUDA OOM on isaaclab_train_rsl_rl-1283 -> fall back to 4 GPU x 1024 envs (same total envs).
NUM_GPUS=4
NUM_ENVS=1024
NUM_CPUS=40
MEMORY=128Gi
TAG_SUFFIX=${TAG_SUFFIX/g2e2048/g4e1024}
