# How to Fix Pyright Errors

## ✅ Environment Issues Fixed

The scikit-learn version conflicts have been resolved by retraining the model in the virtual environment.

## 🔧 Fix Pyright "Undefined Variable" Errors

The errors you're seeing for built-in Python functions like `sorted`, `float`, `dict`, and `round` are Pyright configuration issues. Here's how to fix them:

### Method 1: Restart Language Server (Recommended)

1. **Open Command Palette**: Press `Ctrl+Shift+P`
2. **Type**: "Python: Restart Language Server"
3. **Press Enter**

### Method 2: Select Correct Python Interpreter

1. **Open Command Palette**: Press `Ctrl+Shift+P`
2. **Type**: "Python: Select Interpreter"
3. **Choose**: `./venv/Scripts/python.exe`

### Method 3: Reload Window

1. **Open Command Palette**: Press `Ctrl+Shift+P`
2. **Type**: "Developer: Reload Window"
3. **Press Enter**

### Method 4: Manual Configuration

If the above methods don't work, the configuration files have been updated:

- `pyrightconfig.json` - Pyright configuration
- `.vscode/settings.json` - VS Code Python settings

## 🚀 Running the Application

### Flask App (Web Interface)

```powershell
# Activate virtual environment
venv\Scripts\activate

# Run Flask app
python app.py
```

Then open: http://localhost:5000

### Training Script

```powershell
# Activate virtual environment
venv\Scripts\activate

# Run training
python train_model.py
```

### Using Helper Scripts

```powershell
# Run Flask app
.\run_app.ps1

# Run training
.\run_training.ps1
```

## 📝 Notes

- The Pyright errors are just IDE warnings and don't affect the actual functionality
- The application works perfectly despite these warnings
- All dependencies are properly installed in the virtual environment
- The model has been retrained to fix version conflicts
