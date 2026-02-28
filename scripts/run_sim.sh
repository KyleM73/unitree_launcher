#!/bin/bash
# Launch simulation with MuJoCo viewer.
# macOS: uses mjpython (GLFW main-thread requirement).
# Linux: uses standard python.
# Usage: ./scripts/run_sim.sh --policy path/to/policy.onnx [options]
if [[ "$(uname)" == "Darwin" ]]; then
    exec mjpython -m unitree_launcher.main sim "$@"
else
    exec python -m unitree_launcher.main sim "$@"
fi
