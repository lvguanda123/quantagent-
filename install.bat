@echo off
echo === QuantAgent Setup (Windows) ===

REM 1. Create virtual environment
if not exist "venv" (
    echo [1/4] Creating virtual environment...
    python -m venv venv
) else (
    echo [1/4] Virtual environment already exists.
)

REM 2. Activate
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

REM 3. Install dependencies
echo [3/4] Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM 4. Done
echo [4/4] Setup complete!
echo.
echo Next steps:
echo   1. Activate: venv\Scripts\activate.bat
echo   2. Start:    python web_interface.py
echo   3. Open:     http://127.0.0.1:5000
echo.
pause
