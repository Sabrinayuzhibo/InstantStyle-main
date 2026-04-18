import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

from ip_adapter import IPAdapterPlusXL

# Base SDXL model from Hugging Face Hub.
base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
# Local CLIP image encoder used by IP-Adapter Plus.
image_encoder_path = "models/image_encoder"
# IP-Adapter Plus checkpoint file.
ip_ckpt = "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"
device = "cuda"

# Build the SDXL generation pipeline in fp16 for faster GPU inference.
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
)
# Reduce VRAM usage during VAE decode.
pipe.enable_vae_tiling()

# Inject IP-Adapter Plus into selected attention blocks.
# target_blocks=["block"] for original IP-Adapter
# target_blocks=["up_blocks.0.attentions.1"] for style blocks only
# target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
ip_model = IPAdapterPlusXL(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16, target_blocks=["up_blocks.0.attentions.1"])

# Reference style image.
image = "./assets/0.jpg"
image = Image.open(image)
image.resize((512, 512))

# Run text-to-image generation with style guidance.
images = ip_model.generate(pil_image=image,
                           prompt="a cat, masterpiece, best quality, high quality",
                           negative_prompt= "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                           scale=1.0,
                           guidance_scale=5,
                           num_samples=1,
                           num_inference_steps=30, 
                           seed=42,
                          )

# Save the first generated result.
images[0].save("result.png")