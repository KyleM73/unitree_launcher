#!/bin/bash
# Launch on real robot (onboard G1 only).
# Usage: ./scripts/run_real.sh -c configs/g1_deploy.yaml --policy path/to/policy.onnx [options]
uv run real "$@"
