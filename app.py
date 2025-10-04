#!/usr/bin/env python3
"""
Hugging Face Spaces entry point for IDM-VTON application.
This file serves as the main entry point for the Hugging Face Space.
"""

import sys
import os

# Check Python version first
def check_python_version():
    """Check if Python 3.10 is being used"""
    version = sys.version_info
    if version.major != 3 or version.minor != 10:
        print("❌ ERROR: This application requires Python 3.10")
        print(f"   Current version: Python {version.major}.{version.minor}.{version.micro}")
        print("\n📋 To install Python 3.10:")
        print("   • macOS: brew install python@3.10")
        print("   • Ubuntu: sudo apt install python3.10")
        print("   • Windows: Download from https://www.python.org/downloads/")
        print("\n💡 After installation, use: python3.10 app.py")
        sys.exit(1)
    print(f"✅ Python version check passed: {version.major}.{version.minor}.{version.micro}")

# Check Python version before importing anything else
check_python_version()

# Set environment variables for better compatibility
os.environ['TORCH_HOME'] = '/tmp/torch'
os.environ['HF_HOME'] = '/tmp/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU-only mode

# Fix for various warnings in containers
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add the gradio_demo directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gradio_demo'))

# Import and run the main application
if __name__ == "__main__":
    try:
        print("🔧 Setting up environment...")

        # Try to import torch first to catch PyTorch issues early
        try:
            import torch
            print(f"✅ PyTorch {torch.__version__} loaded successfully")
            print(f"🖥️  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        except Exception as torch_error:
            print(f"⚠️  PyTorch import warning: {torch_error}")
            print("🔄 Continuing with CPU-only mode...")

        # Import the main app from gradio_demo
        print("📦 Loading IDM-VTON application...")
        from gradio_demo.app import demo

        # Launch the Gradio interface
        print("🚀 Starting IDM-VTON Gradio interface...")
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=False
        )
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   python -m pip install -r requirements.txt")
        print("\n🔍 Debugging info:")
        print(f"   Python path: {sys.path}")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Files in current directory: {os.listdir('.')}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
