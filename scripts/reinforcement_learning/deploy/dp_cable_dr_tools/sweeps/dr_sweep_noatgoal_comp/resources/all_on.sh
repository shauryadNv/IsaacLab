# Core baseline: always submitted (and relaunched) at HIGH priority so it is not preempted.
# It queues until the pool quota (32 GPUs for HIGH/NORMAL) has room.
PRIORITY=HIGH
