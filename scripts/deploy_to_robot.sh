#!/bin/bash
# Deploy unitree_launcher to G1 robot via rsync, then run preflight checks.
#
# Usage:
#   ./scripts/deploy_to_robot.sh                          # Default: 192.168.123.164
#   ./scripts/deploy_to_robot.sh 192.168.123.164 unitree  # Explicit IP and user
#
# After deploy, SSH in and run:
#   ssh unitree@192.168.123.164
#   cd ~/unitree_launcher && uv run real --gantry
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

    # Always include ~/.local/bin (where uv installs itself)
    export PATH="$HOME/.local/bin:$PATH"

    # Check uv
    if command -v uv &>/dev/null; then
        echo "OK: uv $(uv --version 2>/dev/null || echo '(version unknown)')"
    else
        echo "uv not found — installing..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"

        # Ensure ~/.local/bin is on PATH for future bash sessions
        PROFILE="$HOME/.bashrc"
        if ! grep -q 'local/bin' "$PROFILE" 2>/dev/null; then
            echo '' >> "$PROFILE"
            echo '# Added by unitree_launcher deploy script' >> "$PROFILE"
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$PROFILE"
            echo "  Added ~/.local/bin to $PROFILE"
        fi

        echo "OK: uv $(uv --version 2>/dev/null || echo '(version unknown)')"
    fi

    # Sync dependencies (--inexact preserves manually installed packages like unitree_cpp)
    uv sync --inexact --quiet 2>/dev/null && echo "OK: uv sync" || echo "WARN: uv sync failed"

    # Check C++ backend
    if uv run python -c "from unitree_cpp import UnitreeController" 2>/dev/null; then
        echo "OK: unitree_cpp"
    else
        echo "WARN: unitree_cpp not installed"
        echo "  Run on robot: cd $ROBOT_DIR && ./scripts/build_cpp_backend.sh"
    fi
PREFLIGHT

echo ""
echo "Deploy complete. To run on robot:"
echo "  ssh $ROBOT_USER@$ROBOT_IP"
echo "  cd $ROBOT_DIR"
echo "  uv run real --gantry                    # gantry arm test"
echo "  uv run real --policy assets/policies/beyondmimic_29dof.onnx  # BeyondMimic"
