#!/bin/bash
# Launch on real robot via DDS.
# Usage: ./scripts/run_real.sh --policy path/to/policy.onnx --interface eth0 [options]
python -m src.main real --config configs/default.yaml "$@"
