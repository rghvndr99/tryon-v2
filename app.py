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

# Add the gradio_demo directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gradio_demo'))

# Import and run the main application
if __name__ == "__main__":
    try:
        # Import the main app from gradio_demo
        from gradio_demo.app import demo
        
        # Launch the Gradio interface
        print("🚀 Starting IDM-VTON Gradio interface...")
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   python -m pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)
