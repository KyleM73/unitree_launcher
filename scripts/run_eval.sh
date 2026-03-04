#!/bin/bash
# Headless batch evaluation (no viewer).
# Usage: ./scripts/run_eval.sh --policy path/to/policy.onnx --steps 500 [options]
uv run eval "$@"
