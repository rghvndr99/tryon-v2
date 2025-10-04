import sys
import os
import warnings

# Comprehensive Hugging Face Spaces compatibility setup
def setup_hf_spaces_environment():
    """Setup environment for Hugging Face Spaces compatibility"""
    print("🔧 Setting up environment for Hugging Face Spaces...")

    # Check Python version
    version = sys.version_info
    if version.major != 3 or version.minor != 10:
        print("❌ ERROR: This application requires Python 3.10")
        print(f"   Current version: Python {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    print(f"✅ Python 3.10 detected: {version.major}.{version.minor}.{version.micro}")

    # Set environment variables for HF Spaces
    os.environ['TORCH_HOME'] = '/tmp/torch'
    os.environ['HF_HOME'] = '/tmp/huggingface'
    os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU-only mode
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid tokenizer warnings

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Add paths
    sys.path.append('./')
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    print("✅ Environment setup complete")

# Run setup
setup_hf_spaces_environment()

from PIL import Image
import gradio as gr
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler,AutoencoderKL
from typing import List

import torch
import os
from transformers import AutoTokenizer
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
# Use preprocessing compatibility layer for HF Spaces
from preprocessing_compat import Parsing, OpenPose
# Use compatibility layer for detectron2
from detectron2_compat import convert_PIL_to_numpy, _apply_exif_orientation
import cv2
from torchvision.transforms.functional import to_pil_image

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True :
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask


base_path = 'yisol/IDM-VTON'
example_path = os.path.join(os.path.dirname(__file__), 'example')

def load_models_safely():
    """Load models with proper error handling for HF Spaces"""
    print("📦 Loading IDM-VTON models...")

    # Determine device and dtype for HF Spaces (CPU-only)
    device = "cpu"
    dtype = torch.float32  # Use float32 for CPU compatibility

    try:
        print("🔄 Loading UNet...")
        unet = UNet2DConditionModel.from_pretrained(
            base_path,
            subfolder="unet",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        unet.requires_grad_(False)
        print("✅ UNet loaded successfully")

        print("🔄 Loading tokenizers...")
        tokenizer_one = AutoTokenizer.from_pretrained(
            base_path,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            base_path,
            subfolder="tokenizer_2",
            revision=None,
            use_fast=False,
        )
        print("✅ Tokenizers loaded successfully")

        print("🔄 Loading scheduler...")
        noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")
        print("✅ Scheduler loaded successfully")

        print("🔄 Loading text encoders...")
        text_encoder_one = CLIPTextModel.from_pretrained(
            base_path,
            subfolder="text_encoder",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            base_path,
            subfolder="text_encoder_2",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        print("✅ Text encoders loaded successfully")

        print("🔄 Loading image encoder...")
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        print("✅ Image encoder loaded successfully")

        print("🔄 Loading VAE...")
        vae = AutoencoderKL.from_pretrained(
            base_path,
            subfolder="vae",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        print("✅ VAE loaded successfully")

        print("🔄 Loading UNet Encoder...")
        UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
            base_path,
            subfolder="unet_encoder",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        print("✅ UNet Encoder loaded successfully")

        print("🔄 Loading preprocessing models...")
        parsing_model = Parsing(0)
        openpose_model = OpenPose(0)
        print("✅ Preprocessing models loaded successfully")

        # Set models to eval mode and disable gradients
        UNet_Encoder.requires_grad_(False)
        image_encoder.requires_grad_(False)
        vae.requires_grad_(False)
        unet.requires_grad_(False)
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)

        return (unet, tokenizer_one, tokenizer_two, noise_scheduler,
                text_encoder_one, text_encoder_two, image_encoder, vae,
                UNet_Encoder, parsing_model, openpose_model)

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("🔄 This might be due to missing model files or network issues")
        print("💡 In HF Spaces, models are downloaded automatically on first run")
        raise e

# Load all models
(unet, tokenizer_one, tokenizer_two, noise_scheduler,
 text_encoder_one, text_encoder_two, image_encoder, vae,
 UNet_Encoder, parsing_model, openpose_model) = load_models_safely()
# Setup transforms and pipeline
tensor_transfrom = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

print("🔄 Creating try-on pipeline...")
pipe = TryonPipeline.from_pretrained(
    base_path,
    unet=unet,
    vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler,
    image_encoder=image_encoder,
    torch_dtype=torch.float32,  # Use float32 for CPU compatibility
)
pipe.unet_encoder = UNet_Encoder

# Set device for HF Spaces (CPU-only)
device = "cpu"
print(f"🖥️  Device: {device}")

# Move models to device
print("🔄 Moving models to device...")
try:
    pipe.to(device)
    pipe.unet_encoder.to(device)
    print("✅ Models moved to device successfully")
except Exception as e:
    print(f"⚠️  Warning: Could not move all models to device: {e}")

print("✅ IDM-VTON pipeline ready!")

def start_tryon(dict, garm_img, garment_des, is_checked, is_checked_crop, denoise_steps, seed):
    """Main try-on function with comprehensive error handling"""
    try:
        print("🔄 Starting virtual try-on process...")

        # Ensure models are on correct device
        if hasattr(openpose_model, 'preprocessor') and hasattr(openpose_model.preprocessor, 'body_estimation'):
            openpose_model.preprocessor.body_estimation.model.to(device)

        # Models should already be on device from initialization

    garm_img= garm_img.convert("RGB").resize((768,1024))
    human_img_orig = dict["background"].convert("RGB")    
    
    if is_checked_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768,1024))
    else:
        human_img = human_img_orig.resize((768,1024))


    if is_checked:
        keypoints = openpose_model(human_img.resize((384,512)))
        model_parse, _ = parsing_model(human_img.resize((384,512)))
        mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
        mask = mask.resize((768,1024))
    else:
        mask = pil_to_binary_mask(dict['layers'][0].convert("RGB").resize((768, 1024)))
        # mask = transforms.ToTensor()(mask)
        # mask = mask.unsqueeze(0)
    mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray+1.0)/2.0)


    human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
     
    

    args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    # verbosity = getattr(args, "verbosity", None)
    pose_img = args.func(args,human_img_arg)    
    pose_img = pose_img[:,:,::-1]    
    pose_img = Image.fromarray(pose_img).resize((768,1024))
    
    with torch.no_grad():
        # Extract the images
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                prompt = "model is wearing " + garment_des
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )
                                    
                    prompt = "a photo of " + garment_des
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                    if not isinstance(prompt, List):
                        prompt = [prompt] * 1
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * 1
                    with torch.inference_mode():
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )



                    pose_img =  tensor_transfrom(pose_img).unsqueeze(0).to(device,torch.float16)
                    garm_tensor =  tensor_transfrom(garm_img).unsqueeze(0).to(device,torch.float16)
                    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
                    images = pipe(
                        prompt_embeds=prompt_embeds.to(device,torch.float16),
                        negative_prompt_embeds=negative_prompt_embeds.to(device,torch.float16),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(device,torch.float16),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device,torch.float16),
                        num_inference_steps=denoise_steps,
                        generator=generator,
                        strength = 1.0,
                        pose_img = pose_img.to(device,torch.float16),
                        text_embeds_cloth=prompt_embeds_c.to(device,torch.float16),
                        cloth = garm_tensor.to(device,torch.float16),
                        mask_image=mask,
                        image=human_img, 
                        height=1024,
                        width=768,
                        ip_adapter_image = garm_img.resize((768,1024)),
                        guidance_scale=2.0,
                    )[0]

    if is_checked_crop:
        out_img = images[0].resize(crop_size)        
        human_img_orig.paste(out_img, (int(left), int(top)))    
        return human_img_orig, mask_gray
    else:
        return images[0], mask_gray
    # return images[0], mask_gray

garm_list = os.listdir(os.path.join(example_path,"cloth"))
garm_list_path = [os.path.join(example_path,"cloth",garm) for garm in garm_list]

human_list = os.listdir(os.path.join(example_path,"human"))
human_list_path = [os.path.join(example_path,"human",human) for human in human_list]

human_ex_list = []
for ex_human in human_list_path:
    ex_dict= {}
    ex_dict['background'] = ex_human
    ex_dict['layers'] = None
    ex_dict['composite'] = None
    human_ex_list.append(ex_dict)

##default human


image_blocks = gr.Blocks().queue()
with image_blocks as demo:
    gr.Markdown("## IDM-VTON 👕👔👚")
    gr.Markdown("Virtual Try-on with your image and garment image. Check out the [source codes](https://github.com/yisol/IDM-VTON) and the [model](https://huggingface.co/yisol/IDM-VTON)")
    with gr.Row():
        with gr.Column():
            imgs = gr.ImageEditor(sources='upload', type="pil", label='Human. Mask with pen or use auto-masking', interactive=True)
            with gr.Row():
                is_checked = gr.Checkbox(label="Yes", info="Use auto-generated mask (Takes 5 seconds)",value=True)
            with gr.Row():
                is_checked_crop = gr.Checkbox(label="Yes", info="Use auto-crop & resizing",value=False)

            example = gr.Examples(
                inputs=imgs,
                examples_per_page=10,
                examples=human_ex_list
            )

        with gr.Column():
            garm_img = gr.Image(label="Garment", sources='upload', type="pil")
            with gr.Row(elem_id="prompt-container"):
                with gr.Row():
                    prompt = gr.Textbox(placeholder="Description of garment ex) Short Sleeve Round Neck T-shirts", show_label=False, elem_id="prompt")
            example = gr.Examples(
                inputs=garm_img,
                examples_per_page=8,
                examples=garm_list_path)
        with gr.Column():
            # image_out = gr.Image(label="Output", elem_id="output-img", height=400)
            masked_img = gr.Image(label="Masked image output", elem_id="masked-img",show_share_button=False)
        with gr.Column():
            # image_out = gr.Image(label="Output", elem_id="output-img", height=400)
            image_out = gr.Image(label="Output", elem_id="output-img",show_share_button=False)




    with gr.Column():
        try_button = gr.Button(value="Try-on")
        with gr.Accordion(label="Advanced Settings", open=False):
            with gr.Row():
                denoise_steps = gr.Number(label="Denoising Steps", minimum=20, maximum=40, value=30, step=1)
                seed = gr.Number(label="Seed", minimum=-1, maximum=2147483647, step=1, value=42)



    try_button.click(fn=start_tryon, inputs=[imgs, garm_img, prompt, is_checked,is_checked_crop, denoise_steps, seed], outputs=[image_out,masked_img], api_name='tryon')

            


# Export the demo object for use by other scripts
demo = image_blocks

# Only launch if this script is run directly
if __name__ == "__main__":
    image_blocks.launch()

