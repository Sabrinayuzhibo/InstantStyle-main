import os
import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import torch
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
    parser = argparse.ArgumentParser(description="SD1.5 + IP-Adapter + ControlNet(Canny) inference")
    parser.add_argument(
        "--config",
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
controlnet_path = paths_cfg.get("controlnet_path", "diffusers_models/control_v11p_sd15_canny")
image_encoder_path = paths_cfg.get("image_encoder_path", "models/image_encoder")
ip_ckpt = paths_cfg.get("ip_ckpt", "models/ip-adapter_sd15.bin")
style_image_path = paths_cfg.get("style_image_path", "test_ref_images/blue.jpg")
control_image_path = paths_cfg.get("control_image_path", "test_images/004.jpg")
output_dir_cfg = paths_cfg.get("output_dir")
legacy_output_image_path = paths_cfg.get("output_image_path", "result_sd15_controlnet.png")
legacy_output_image_path = Path(legacy_output_image_path)


def _infer_control_type(controlnet_path_value: str) -> str:
    name = Path(controlnet_path_value).name.lower()
    if "canny" in name:
        return "canny"
    if "lineart" in name:
        return "lineart"
    return name.replace(" ", "_")


def _fmt_num(v: float) -> str:
    return f"{v:g}"

device = _get_device(runtime_cfg.get("device") or os.environ.get("INSTANTSTYLE_DEVICE"))
torch_dtype = torch.float16 if device == "cuda" else torch.float32

canny_low_threshold = int(preprocess_cfg.get("canny_low_threshold", 50))
canny_high_threshold = int(preprocess_cfg.get("canny_high_threshold", 200))
bilateral_d = int(preprocess_cfg.get("bilateral_d", 0))
bilateral_sigma_color = float(preprocess_cfg.get("bilateral_sigma_color", 50))
bilateral_sigma_space = float(preprocess_cfg.get("bilateral_sigma_space", 50))
canny_aperture_size = int(preprocess_cfg.get("canny_aperture_size", 3))
canny_l2gradient = bool(preprocess_cfg.get("canny_l2gradient", False))
use_canny_cache = bool(preprocess_cfg.get("use_canny_cache", True))
canny_cache_path_cfg = preprocess_cfg.get("canny_cache_path")

if canny_aperture_size not in {3, 5, 7}:
    raise ValueError("canny_aperture_size must be one of: 3, 5, 7")

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
adain_ip = bool(ip_adapter_cfg.get("adain_ip", False))
adain_alpha = float(ip_adapter_cfg.get("adain_alpha", 1.0))
adain_beta = float(ip_adapter_cfg.get("adain_beta", 1.0))
controlnet_variant = model_loading_cfg.get("controlnet_variant", "fp16")
controlnet_use_safetensors = bool(model_loading_cfg.get("controlnet_use_safetensors", False))

# Auto-name output file using current runtime configuration:
# res_{canny还是其他的}_controlent_{noadain/adain}_{input}_scale{ }_cond_scale{ }
control_type = _infer_control_type(controlnet_path)
adain_tag = "adain" if adain_ip else "noadain"
control_image_stem = Path(control_image_path).stem
auto_output_name = (
    f"res_{control_type}_controlent_{adain_tag}_{control_image_stem}"
    f"_scale{_fmt_num(scale)}_cond_scale{_fmt_num(controlnet_conditioning_scale)}.png"
)
if output_dir_cfg:
    output_base_dir = Path(str(output_dir_cfg)).expanduser()
else:
    # Legacy compatibility: derive output directory from paths.output_image_path
    output_base_dir = (
        legacy_output_image_path.parent if legacy_output_image_path.suffix else legacy_output_image_path
    )
output_image_path = output_base_dir / auto_output_name

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

ip_model = IPAdapter(
    pipe,
    str(image_encoder_dir),
    str(ip_ckpt_path),
    device,
    target_blocks=target_blocks,
    adain_ip=adain_ip,
    adain_alpha=adain_alpha,
    adain_beta=adain_beta,
)

# Debug report: verify AdaIN-enabled processors before running inference.
adain_enabled_processors = []
all_attn_processor_names = list(ip_model.pipe.unet.attn_processors.keys())
for proc_name, proc in ip_model.pipe.unet.attn_processors.items():
    if getattr(proc, "adainIP", False):
        adain_enabled_processors.append(proc_name)
        if hasattr(proc, "adain_call_count"):
            proc.adain_call_count = 0

print(f"[debug] adain_ip_config={adain_ip}, adain_enabled_processor_count={len(adain_enabled_processors)}")
if adain_enabled_processors:
    print("[debug] sample_adain_processors:", adain_enabled_processors[:5])
elif adain_ip:
    print("[warn] AdaIN is enabled but no attention processor matched target_blocks.")
    print("[warn] sample_unet_attn_processors:", all_attn_processor_names[:8])
    print("[hint] SD1.5 commonly uses target prefixes like: down_blocks.2, mid_block, up_blocks.1")

# style/reference image for IP-Adapter
style_image_path = _require_exists(Path(style_image_path), "Style image")
style_image = Image.open(style_image_path).convert("RGB")

# canny control image for ControlNet
control_image_path = _require_exists(Path(control_image_path), "Control image")
input_image = cv2.imread(str(control_image_path))
if input_image is None:
    raise FileNotFoundError(f"Control image not found or unreadable: {control_image_path}")

if canny_cache_path_cfg:
    configured_cache_path = Path(str(canny_cache_path_cfg))
    # If a directory is provided (e.g. "canny_cache/"), auto-name as <input_stem>_canny.png.
    if str(canny_cache_path_cfg).endswith(("/", "\\")) or configured_cache_path.suffix == "":
        canny_cache_path = configured_cache_path / f"{control_image_path.stem}_canny.png"
    else:
        canny_cache_path = configured_cache_path
else:
    canny_cache_path = Path("cache") / (
        f"{control_image_path.stem}_canny.png"
    )

canny_signature = (
    f"low={canny_low_threshold};high={canny_high_threshold};ap={canny_aperture_size};"
    f"l2={int(canny_l2gradient)};bd={bilateral_d};bsc={bilateral_sigma_color};bss={bilateral_sigma_space}"
)
canny_meta_path = canny_cache_path.with_suffix(canny_cache_path.suffix + ".meta")

use_cached_canny = False
if use_canny_cache and canny_cache_path.exists():
    cache_newer_than_source = canny_cache_path.stat().st_mtime >= control_image_path.stat().st_mtime
    signature_ok = canny_meta_path.exists() and canny_meta_path.read_text(encoding="utf-8").strip() == canny_signature
    if cache_newer_than_source and signature_ok:
        use_cached_canny = True

if use_cached_canny:
    canny_map = Image.open(canny_cache_path).convert("RGB")
else:
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    if bilateral_d > 0:
        gray_image = cv2.bilateralFilter(
            gray_image,
            d=bilateral_d,
            sigmaColor=bilateral_sigma_color,
            sigmaSpace=bilateral_sigma_space,
        )
    detected_map = cv2.Canny(
        gray_image,
        canny_low_threshold,
        canny_high_threshold,
        apertureSize=canny_aperture_size,
        L2gradient=canny_l2gradient,
    )
    canny_map = Image.fromarray(detected_map).convert("RGB")

# Always export the canny control map for the current inference run.
# This guarantees a visible canny image in cache directory every time.
canny_cache_path.parent.mkdir(parents=True, exist_ok=True)
canny_map.save(canny_cache_path)
canny_meta_path.write_text(canny_signature, encoding="utf-8")

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
    image=canny_map,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
)

adain_total_calls = 0
for proc in ip_model.pipe.unet.attn_processors.values():
    adain_total_calls += int(getattr(proc, "adain_call_count", 0))
print(f"[debug] adain_runtime_call_count={adain_total_calls}")

# Organize each run output into a folder named after image filename (without extension).
output_dir = output_image_path.parent / output_image_path.stem
output_dir.mkdir(parents=True, exist_ok=True)

final_image_path = output_dir / output_image_path.name
images[0].save(final_image_path)

# Save the exact config used for this generation alongside the image.
config_copy_path = output_dir / "used_config.yaml"
shutil.copy2(config_path, config_copy_path)
