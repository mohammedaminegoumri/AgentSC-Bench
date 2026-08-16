#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "=== AgentSC-Bench smoke test ==="
python -m experiments.runner --config configs/baseline.yaml --output results/smoke
python -m experiments.runner --config configs/mock_llm.yaml --output results/smoke
pytest -q
echo "=== Smoke test completed successfully ==="
