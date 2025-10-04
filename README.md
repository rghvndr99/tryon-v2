
---
title: Tryon V2
emoji: 👀
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 3.50.2
app_file: app.py
pinned: false
license: apache-2.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# IDM-VTON: Virtual Try-On Application

<div align="center">

<a href='https://idm-vton.github.io'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/abs/2403.05139'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href='https://huggingface.co/spaces/yisol/IDM-VTON'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-yellow'></a>
<a href='https://huggingface.co/yisol/IDM-VTON'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>

</div>

Virtual try-on application using diffusion models for authentic garment fitting.

![teaser2](assets/teaser2.png)

## Requirements

**⚠️ IMPORTANT: This application requires Python 3.10 exactly**

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

## Model Checkpoints

You need to download the model checkpoints to run the application:

1. **Download from Hugging Face:**
   ```bash
   # Download the main model
   git clone https://huggingface.co/yisol/IDM-VTON ckpt/
   ```

2. **Or download individual components:**
   - DensePose model: `ckpt/densepose/model_final_162be9.pkl`
   - Human parsing models: `ckpt/humanparsing/`
   - OpenPose model: `ckpt/openpose/`
   - IP-Adapter: `ckpt/ip_adapter/`

## Usage

1. **Start the Gradio interface:**
   ```bash
   python gradio_demo/app.py
   ```

2. **Upload images:**
   - Upload a person image
   - Upload a garment image
   - The app will automatically generate the try-on result

## Features

- ✅ Automatic pose estimation
- ✅ Human parsing and segmentation
- ✅ Garment-person alignment
- ✅ High-quality diffusion-based try-on
- ✅ Web interface with Gradio

## Troubleshooting

### Common Issues

1. **Python Version Error:**
   - Ensure you're using Python 3.10 exactly
   - Use `python --version` to check

2. **Missing Checkpoints:**
   - Download all required model files to `ckpt/` directory
   - Check the Hugging Face repository for complete files

3. **CUDA/GPU Issues:**
   - The app will automatically use CPU if CUDA is not available
   - For better performance, ensure CUDA is properly installed

## Acknowledgements

Thanks to the original [IDM-VTON](https://github.com/yisol/IDM-VTON) authors and contributors.

## License

This project is under the [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).


