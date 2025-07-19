Write-Host "Activating virtual environment..." -ForegroundColor Green
& "venv\Scripts\Activate.ps1"
Write-Host "Starting Flask app..." -ForegroundColor Green
python app.py 