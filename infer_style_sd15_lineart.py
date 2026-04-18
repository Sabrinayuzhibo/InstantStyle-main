import os
import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import torch
from controlnet_aux import LineartDetector
from diffusers.models.controlnets.controlnet import ControlNetModel
from diffusers.pipelines.controlnet.pipeline_controlnet import StableDiffusionControlNetPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from PIL import Image
import yaml

from ip_adapter import IPAdapter


def _get_device(requested: Optional[str]) -> str:
    if requested:
        requested = requested.strip().lower()
    if requested in {"cuda", "cpu"}:
        if requested == "cuda" and not torch.cuda.is_available():
            print("[warn] device=cuda requested but CUDA is not available; falling back to cpu")
            return "cpu"
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _require_exists(path: Path, what: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


def _load_config(config_path: Path) -> Dict[str, Any]:
    _require_exists(config_path, "Config file")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SD1.5 + IP-Adapter + ControlNet(Lineart) inference")
    parser.add_argument(
        "--config",
        "--confiig",
        dest="config",
        type=str,
        default=None,
        help="Path to YAML config file. Defaults to config.yaml next to this script.",
    )
    return parser.parse_args()


args = _parse_args()
config_path = Path(args.config).expanduser() if args.config else Path(__file__).resolve().with_name("config.yaml")
cfg = _load_config(config_path)

paths_cfg = cfg.get("paths", {})
runtime_cfg = cfg.get("runtime", {})
preprocess_cfg = cfg.get("preprocess", {})
generate_cfg = cfg.get("generate", {})
output_cfg = cfg.get("output", {})
ip_adapter_cfg = cfg.get("ip_adapter", {})
model_loading_cfg = cfg.get("model_loading", {})

base_model_path = paths_cfg.get("base_model_path", "D:/AI_Models/stable-diffusion-v1-5")
controlnet_path = paths_cfg.get("controlnet_path", "diffusers_models/control_v11p_sd15_lineart")
image_encoder_path = paths_cfg.get("image_encoder_path", "models/image_encoder")
ip_ckpt = paths_cfg.get("ip_ckpt", "models/ip-adapter_sd15.bin")
style_image_path = paths_cfg.get("style_image_path", "test_ref_images/blue.jpg")
control_image_path = paths_cfg.get("control_image_path", "test_images/004.jpg")
output_image_path = paths_cfg.get("output_image_path", "result_sd15_controlnet.png")

device = _get_device(runtime_cfg.get("device") or os.environ.get("INSTANTSTYLE_DEVICE"))
torch_dtype = torch.float16 if device == "cuda" else torch.float32

lineart_coarse = bool(preprocess_cfg.get("lineart_coarse", False))
lineart_detector_model = str(preprocess_cfg.get("lineart_detector_model", "lllyasviel/Annotators"))
use_lineart_cache = bool(preprocess_cfg.get("use_lineart_cache", True))
lineart_cache_path_cfg = preprocess_cfg.get("lineart_cache_path")

prompt = str(
    generate_cfg.get(
        "prompt",
        "A beautiful woman, close-up portrait, hand near face, looking at viewer, watercolor painting style, soft pastel colors, light blue tones, aesthetic illustration, masterpiece, best quality, highly detailed",
    )
)
negative_prompt = str(
    generate_cfg.get(
        "negative_prompt",
        "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry, ugly, bad anatomy",
    )
)
scale = float(generate_cfg.get("scale", 0.2))
guidance_scale = float(generate_cfg.get("guidance_scale", 5))
num_samples = int(generate_cfg.get("num_samples", 1))
num_inference_steps = int(generate_cfg.get("num_inference_steps", 20))
seed = int(generate_cfg.get("seed", 42))
controlnet_conditioning_scale = float(generate_cfg.get("controlnet_conditioning_scale", 0.7))

use_control_image_size = bool(output_cfg.get("use_control_image_size", True))
configured_width = output_cfg.get("width")
configured_height = output_cfg.get("height")
target_blocks = ip_adapter_cfg.get("target_blocks", ["block"])
controlnet_variant = model_loading_cfg.get("controlnet_variant", "fp16")
controlnet_use_safetensors = bool(model_loading_cfg.get("controlnet_use_safetensors", False))

base_model_dir = _require_exists(Path(base_model_path).expanduser(), "SD1.5 base model directory")
if not (base_model_dir / "model_index.json").exists():
    raise FileNotFoundError(
        "SD1.5 base model directory does not look like a Diffusers model (missing model_index.json): "
        + str(base_model_dir)
    )

controlnet_dir = _require_exists(Path(controlnet_path).expanduser(), "ControlNet directory")
image_encoder_dir = _require_exists(Path(image_encoder_path).expanduser(), "Image encoder directory")
ip_ckpt_path = _require_exists(Path(ip_ckpt).expanduser(), "IP-Adapter checkpoint")

controlnet = ControlNetModel.from_pretrained(
    str(controlnet_dir),
    variant=controlnet_variant,
    use_safetensors=controlnet_use_safetensors,
    torch_dtype=torch_dtype,
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    str(base_model_dir),
    controlnet=controlnet,
    torch_dtype=torch_dtype,
    safety_checker=None,
    requires_safety_checker=False,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.vae.enable_tiling()

ip_model = IPAdapter(pipe, str(image_encoder_dir), str(ip_ckpt_path), device, target_blocks=target_blocks)

# style/reference image for IP-Adapter
style_image_path = _require_exists(Path(style_image_path), "Style image")
style_image = Image.open(style_image_path).convert("RGB")

# lineart control image for ControlNet
control_image_path = _require_exists(Path(control_image_path), "Control image")
input_image = cv2.imread(str(control_image_path))
if input_image is None:
    raise FileNotFoundError(f"Control image not found or unreadable: {control_image_path}")

if lineart_cache_path_cfg:
    configured_cache_path = Path(str(lineart_cache_path_cfg))
    # If a directory is provided (e.g. "lineart_cache/"), auto-name as <input_stem>_lineart.png.
    if str(lineart_cache_path_cfg).endswith(("/", "\\")) or configured_cache_path.suffix == "":
        lineart_cache_path = configured_cache_path / f"{control_image_path.stem}_lineart.png"
    else:
        lineart_cache_path = configured_cache_path
else:
    lineart_cache_path = Path("cache") / (
        f"{control_image_path.stem}_lineart.png"
    )

use_cached_lineart = False
if use_lineart_cache and lineart_cache_path.exists():
    cache_newer_than_source = lineart_cache_path.stat().st_mtime >= control_image_path.stat().st_mtime
    if cache_newer_than_source:
        use_cached_lineart = True

if use_cached_lineart:
    lineart_map = Image.open(lineart_cache_path).convert("RGB")
else:
    control_pil = Image.open(control_image_path).convert("RGB")
    lineart_detector = LineartDetector.from_pretrained(lineart_detector_model)
    lineart_map = lineart_detector(control_pil, coarse=lineart_coarse).convert("RGB")
    if use_lineart_cache:
        lineart_cache_path.parent.mkdir(parents=True, exist_ok=True)
        lineart_map.save(lineart_cache_path)

control_height, control_width = input_image.shape[:2]

if use_control_image_size:
    out_width, out_height = control_width, control_height
else:
    if configured_width is None or configured_height is None:
        raise ValueError("When use_control_image_size is false, width and height must be set in config.yaml")
    out_width, out_height = int(configured_width), int(configured_height)

images = ip_model.generate(
    pil_image=style_image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    scale=scale,
    guidance_scale=guidance_scale,
    num_samples=num_samples,
    num_inference_steps=num_inference_steps,
    seed=seed,
    width=out_width,
    height=out_height,
    image=lineart_map,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
)
images[0].save(output_image_path)
