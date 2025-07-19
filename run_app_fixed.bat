@echo off
echo Starting Phone Price Prediction App...
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if activation was successful
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

echo Virtual environment activated successfully
echo.

REM Run the fixed application
python app_fixed.py

pause 