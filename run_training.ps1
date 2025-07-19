Write-Host "Activating virtual environment..." -ForegroundColor Green
& "venv\Scripts\Activate.ps1"
Write-Host "Running training script..." -ForegroundColor Green
python train_model.py
Read-Host "Press Enter to continue" 