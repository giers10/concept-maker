#!/usr/bin/env bash
set -euo pipefail

# Resolve to repo root (directory of this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Pick a Python executable
if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

echo "[setup] Creating virtual environment in .venv"
"$PY" -m venv .venv

# Activate venv
echo "[setup] Activating virtual environment"
source .venv/bin/activate

# Upgrade pip (optional but helpful)
python -m pip install --upgrade pip >/dev/null 2>&1 || true

# Install requirements
echo "[deps] Installing requirements from requirements.txt"
python -m pip install -r requirements.txt

# Launch the app
echo "[run] Starting Concept Maker GUI"
exec python concept-maker_gui.py

