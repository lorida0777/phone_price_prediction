Write-Host "Starting Phone Price Prediction App..." -ForegroundColor Green
Write-Host ""

# Activate virtual environment
try {
    & "venv\Scripts\Activate.ps1"
    Write-Host "Virtual environment activated successfully" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "Error: Failed to activate virtual environment" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Run the fixed application
try {
    python app_fixed.py
} catch {
    Write-Host "Error running application:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

Read-Host "Press Enter to exit" 