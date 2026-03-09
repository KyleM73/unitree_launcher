#!/bin/bash
# Sync logs from G1 robot back to this machine.
#
# Usage:
#   ./scripts/get_logs_from_robot.sh                          # Default: 192.168.123.164
#   ./scripts/get_logs_from_robot.sh 192.168.123.164 unitree  # Explicit IP and user
#
# Logs are synced into the local logs/ directory, preserving the same
# directory structure as on the robot. Existing local logs are not deleted.

set -euo pipefail

ROBOT_IP="${1:-192.168.123.164}"
ROBOT_USER="${2:-unitree}"
ROBOT_DIR="/home/$ROBOT_USER/unitree_launcher"
LOCAL_LOGS="./logs/"

mkdir -p "$LOCAL_LOGS"

echo "Fetching logs from $ROBOT_USER@$ROBOT_IP:$ROBOT_DIR/logs/ ..."
rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    "$ROBOT_USER@$ROBOT_IP:$ROBOT_DIR/logs/" "$LOCAL_LOGS"

echo ""
echo "Logs synced to $LOCAL_LOGS"
echo ""
echo "Replay a run:"
echo "  uv run replay logs/<run_name>/ --gui"
echo "  uv run replay logs/<run_name>/ --viser"
