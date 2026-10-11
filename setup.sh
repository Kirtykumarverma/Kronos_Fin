#!/usr/bin/env bash
# Create a virtualenv and install Kronos_Fin dependencies.
# Usage: ./setup.sh            (default PyTorch build for your platform)
#        ./setup.sh --cpu      (smaller CPU-only PyTorch on Linux/Windows)
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

if [[ "${1:-}" == "--cpu" ]]; then
  pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
pip install -r requirements.txt pytest

echo
echo "Done. Activate with: source .venv/bin/activate"
echo "Offline check:       pytest tests/test_smoke_offline.py"
echo "Real forecast:       python run_prediction.py"
