#!/bin/bash
# Launch simulation with MuJoCo viewer.
# Usage: ./scripts/run_sim.sh --policy path/to/policy.onnx [options]
python -m src.main sim --config configs/default.yaml "$@"
