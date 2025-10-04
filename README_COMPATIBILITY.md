# IDM-VTON: Virtual Try-On Application

This is a cleaned-up version of IDM-VTON optimized for production use.

## Requirements

**IMPORTANT: This application requires Python 3.10 exactly**

## Quick Setup

```bash
# Clone the repository
git clone https://github.com/rghvndr99/tryon-v2.git
cd tryon-v2

# Run setup (includes Python version check)
python setup.py

# Run application
python gradio_demo/app.py
```

## Key Features

- Virtual try-on with diffusion models
- Gradio web interface
- Automatic preprocessing (pose estimation, human parsing)
- Compatible with diffusers 0.21.4 and PyTorch 1.12.1
- CUDA-capable GPU (recommended)
- ~15GB disk space for models
- Conda/Miniconda

## Usage

Access the web interface at http://localhost:7860 after running the application.
Upload a person image and garment image to test virtual try-on functionality.
