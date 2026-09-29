# Core baseline: always submitted (and relaunched) at HIGH priority so it is not preempted.
# It queues until the pool quota (32 GPUs for HIGH/NORMAL) has room.
PRIORITY=HIGH

# 2026-09-27T18:17:31Z monitor: CUDA OOM on isaaclab_train_rsl_rl-1230 -> fall back to 4 GPU x 1024 envs (same total envs).
NUM_GPUS=4
NUM_ENVS=1024
NUM_CPUS=40
MEMORY=128Gi
TAG_SUFFIX=${TAG_SUFFIX/g2e2048/g4e1024}
