#!/usr/bin/env bash
# =============================================================================
# setup.sh — Creates venv and installs dependencies for the RAG System
# Usage: bash setup.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo " RAG System — Setup"
echo "========================================"

# 1. Create virtual environment
if [ ! -d "venv" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/4] venv already exists, reusing..."
fi

# 2. Activate venv
echo "[2/4] Activating venv..."
source venv/bin/activate

# 3. Install dependencies
echo "[3/4] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Copy .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "[4/4] Copying .env.example -> .env (edit with your API keys)"
    cp .env.example .env
else
    echo "[4/4] .env already exists, not overwriting."
fi

echo ""
echo "========================================"
echo " Setup complete."
echo " Next steps:"
echo "   1. Edit .env with your API keys"
echo "   2. Ensure PostgreSQL+pgvector is running"
echo "   3. python -m src.indexer       (index data)"
echo "   4. python -m src.web_app       (start web UI)"
echo "   5. python -m tests.run_evaluation (run tests)"
echo "========================================"
