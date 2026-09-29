#!/usr/bin/env bash
# One scheduled tick (cron runs this every 15 minutes):
#   1. make sure the dashboard server is up (http://localhost:8765)
#   2. run the status monitor (osmo status, relaunches, logs, metrics, checkpoint sync)
#   3. start the rollout worker in the background if it is not already running
# Everything logs to monitor/logs/. Settings come from ../env.sh.
set -u
HOME_DIR=$(cd "$(dirname "$0")" && pwd)
export PATH=/usr/local/bin:/usr/bin:/bin:$PATH   # cron has a minimal PATH; osmo is usually in /usr/local/bin
source "$HOME_DIR/../env.sh"
PY=$ISAACLAB_PYTHON
PORT=${MONITOR_PORT:-8765}
mkdir -p "$HOME_DIR/logs"
cd "$HOME_DIR" || exit 1

if ! (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then
  setsid nohup "$PY" -m http.server "$PORT" --bind 127.0.0.1 --directory "$HOME_DIR" \
    > "$HOME_DIR/logs/server.log" 2>&1 < /dev/null &
  echo "$(date -u +%FT%TZ) started dashboard server on :$PORT" >> "$HOME_DIR/logs/tick.log"
fi

"$PY" monitor.py >> "$HOME_DIR/logs/tick.log" 2>&1

# The worker holds its own lock, so a second copy exits immediately.
setsid nohup "$PY" rollout_worker.py >> "$HOME_DIR/logs/rollout.log" 2>&1 < /dev/null &

# Keep logs bounded.
for f in "$HOME_DIR/logs/tick.log" "$HOME_DIR/logs/rollout.log"; do
  [ -f "$f" ] && [ "$(stat -c %s "$f")" -gt 5000000 ] && tail -c 2000000 "$f" > "$f.tmp" && mv "$f.tmp" "$f"
done
exit 0
