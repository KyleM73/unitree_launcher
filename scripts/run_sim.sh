#!/bin/bash
# Launch simulation.
# macOS + --gui: uses mjpython for GLFW main-thread requirement.
# All other cases: uses uv run.
# Usage: ./scripts/run_sim.sh --gui --policy path/to/policy.onnx [options]
if [[ "$(uname)" == "Darwin" ]] && [[ "$*" == *"--gui"* ]]; then
    exec mjpython -m unitree_launcher.main sim "$@"
else
    exec uv run sim "$@"
fi
