@echo off
echo Stopping any existing Python processes...
taskkill /f /im python.exe 2>nul
echo.
echo Activating virtual environment...
call venv\Scripts\activate
echo.
echo Starting Flask app from virtual environment...
python app.py
pause 