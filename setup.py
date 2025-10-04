#!/usr/bin/env python3
"""
IDM-VTON Setup Script
Checks Python version and installs dependencies
"""

import sys
import subprocess
import os
from pathlib import Path

def check_python_version():
    """Check if Python 3.10 is being used"""
    version = sys.version_info
    
    if version.major != 3 or version.minor != 10:
        print("❌ ERROR: This application requires Python 3.10")
        print(f"   Current version: Python {version.major}.{version.minor}.{version.micro}")
        print("\n📋 To fix this:")
        print("   1. Install Python 3.10 from https://www.python.org/downloads/")
        print("   2. Or use pyenv: pyenv install 3.10.12 && pyenv local 3.10.12")
        print("   3. Or use conda: conda create -n idm python=3.10 && conda activate idm")
        print("\n   Then run this setup script again.")
        sys.exit(1)
    
    print(f"✅ Python version check passed: {version.major}.{version.minor}.{version.micro}")

def install_requirements():
    """Install required packages"""
    requirements_file = "requirements.txt"
    
    if not os.path.exists(requirements_file):
        print(f"❌ ERROR: {requirements_file} not found")
        sys.exit(1)
    
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", requirements_file
        ])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Failed to install requirements: {e}")
        sys.exit(1)

def check_directories():
    """Check if essential directories exist"""
    essential_dirs = [
        "src",
        "gradio_demo", 
        "ckpt",
        "preprocess"
    ]
    
    missing_dirs = []
    for dir_name in essential_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"❌ ERROR: Missing essential directories: {', '.join(missing_dirs)}")
        print("   Please ensure you have the complete repository")
        sys.exit(1)
    
    print("✅ Essential directories check passed")

def main():
    """Main setup function"""
    print("🚀 IDM-VTON Setup Starting...")
    print("=" * 50)
    
    # Check Python version first
    check_python_version()
    
    # Check essential directories
    check_directories()
    
    # Install requirements
    install_requirements()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("   1. Download model checkpoints to ./ckpt/ directory")
    print("   2. Run the application: python gradio_demo/app.py")
    print("\n🔗 For model downloads, see: https://github.com/yisol/IDM-VTON")

if __name__ == "__main__":
    main()
