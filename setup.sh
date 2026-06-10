#!/bin/bash
# setup.sh — First-time setup for Multimodal Offline RAG
# Run once: bash setup.sh

set -e

echo "======================================"
echo "  Multimodal Offline RAG — Setup"
echo "======================================"

# ── Python dependencies ───────────────────────────────────────────────────────
echo ""
echo "[1/4] Installing Python dependencies..."
pip install -r requirements.txt

# ── Tesseract OCR ─────────────────────────────────────────────────────────────
echo ""
echo "[2/4] Installing Tesseract OCR..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get update -q && sudo apt-get install -y tesseract-ocr
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install tesseract
elif [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "cygwin"* ]]; then
    echo "  Windows: Install Tesseract from:"
    echo "  https://github.com/UB-Mannheim/tesseract/wiki"
    echo "  Then add to PATH"
fi

# ── Ollama ────────────────────────────────────────────────────────────────────
echo ""
echo "[3/4] Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "  Ollama not found. Installing..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo "  Download Ollama from: https://ollama.com/download"
        echo "  Then re-run this script."
        exit 1
    fi
else
    echo "  Ollama already installed ✓"
fi

# ── Pull LLM model ────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Pulling LLM model (llama3.2:3b — ~2GB, one-time download)..."
ollama pull llama3.2:3b

# ── Create directories ────────────────────────────────────────────────────────
mkdir -p uploads vectordb

echo ""
echo "======================================"
echo "  Setup complete! How to run:"
echo "======================================"
echo ""
echo "  Terminal 1 — Start Ollama:"
echo "    ollama serve"
echo ""
echo "  Terminal 2 — Start API backend:"
echo "    python api.py"
echo ""
echo "  Terminal 3 — Start UI:"
echo "    python ui.py"
echo ""
echo "  Then open: http://localhost:7860"
echo "======================================"
