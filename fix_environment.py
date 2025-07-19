#!/usr/bin/env python3
"""
Script to fix environment issues and retrain the model
"""
import sys
import subprocess
import os

def check_environment():
    """Check if we're in the correct virtual environment"""
    print("Checking Python environment...")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # Check if we're in the virtual environment
    if "venv" in sys.executable:
        print("✅ Running in virtual environment")
        return True
    else:
        print("❌ Not running in virtual environment")
        print("Please activate the virtual environment first:")
        print("  venv\\Scripts\\activate")
        return False

def install_dependencies():
    """Install/upgrade dependencies"""
    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def retrain_model():
    """Retrain the model to fix version conflicts"""
    print("\nRetraining model...")
    try:
        subprocess.check_call([sys.executable, "train_model.py"])
        print("✅ Model retrained successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to retrain model: {e}")
        return False

def main():
    print("🔧 Phone Price Prediction - Environment Fix Script")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Retrain model
    if not retrain_model():
        return False
    
    print("\n🎉 Environment fixed successfully!")
    print("You can now run the Flask app with:")
    print("  python app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 