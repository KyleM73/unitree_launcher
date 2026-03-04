#!/bin/bash
# Deploy unitree_launcher to G1 robot via rsync, then run preflight checks.
#
# Usage:
#   ./scripts/deploy_to_robot.sh                          # Default: 192.168.123.164
#   ./scripts/deploy_to_robot.sh 192.168.123.164 unitree  # Explicit IP and user
#
# After deploy, SSH in and run:
#   ssh unitree@192.168.123.164
#   cd ~/unitree_launcher && uv run real -c configs/g1_deploy.yaml
#
# To build the C++ backend on the robot (first time only):
#   ./scripts/build_cpp_backend.sh

set -euo pipefail

ROBOT_IP="${1:-192.168.123.164}"
ROBOT_USER="${2:-unitree}"
ROBOT_DIR="/home/$ROBOT_USER/unitree_launcher"

echo "Syncing to $ROBOT_USER@$ROBOT_IP:$ROBOT_DIR ..."
rsync -avz --delete \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '.venv' \
    --exclude '.mypy_cache' \
    --exclude '.pytest_cache' \
    --exclude 'reference' \
    --exclude 'refactor' \
    --exclude 'tests' \
    --exclude 'docker' \
    --exclude 'logs' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    ./ "$ROBOT_USER@$ROBOT_IP:$ROBOT_DIR/"

echo ""
echo "Running preflight checks on robot..."
ssh "$ROBOT_USER@$ROBOT_IP" bash -s "$ROBOT_DIR" <<'PREFLIGHT'
    ROBOT_DIR="$1"
    cd "$ROBOT_DIR"

    # Check uv — install if missing
    if ! command -v uv &>/dev/null; then
        echo "uv not found — installing..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    echo "OK: uv $(uv --version 2>/dev/null || echo '(version unknown)')"

    # Sync dependencies
    uv sync --quiet 2>/dev/null && echo "OK: uv sync" || echo "WARN: uv sync failed"

    # Check C++ backend
    if uv run python -c "import unitree_interface" 2>/dev/null; then
        echo "OK: unitree_interface"
    else
        echo "WARN: unitree_interface not installed"
        echo "  Run on robot: cd $ROBOT_DIR && ./scripts/build_cpp_backend.sh"
    fi
PREFLIGHT

echo ""
echo "Deploy complete. To run on robot:"
echo "  ssh $ROBOT_USER@$ROBOT_IP"
echo "  cd $ROBOT_DIR"
echo "  uv run real -c configs/g1_deploy.yaml"
