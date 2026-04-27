#!/usr/bin/env bash
# QuantAgent - Linux/macOS Setup Script
set -e

echo "=== QuantAgent Setup ==="

# 1. Create virtual environment
if [ ! -d "venv" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/4] Virtual environment already exists."
fi

# 2. Activate
echo "[2/4] Activating virtual environment..."
source venv/bin/activate

# 3. Install dependencies
echo "[3/4] Installing dependencies..."
pip install --upgrade pip

# TA-Lib may need the C library first
# Ubuntu/Debian: sudo apt-get install -y ta-lib
# macOS: brew install ta-lib

pip install -r requirements.txt

# 4. Done
echo "[4/4] Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate: source venv/bin/activate"
echo "  2. Start:    python web_interface.py"
echo "  3. Open:     http://127.0.0.1:5000"
