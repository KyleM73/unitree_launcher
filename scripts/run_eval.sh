#!/bin/bash
# Headless batch evaluation (no viewer).
# Usage: ./scripts/run_eval.sh --policy path/to/policy.onnx --duration 30 [options]
python -m unitree_launcher.main sim --headless "$@"
