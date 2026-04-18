import os
import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from diffusers.models.controlnets.controlnet import ControlNetModel
from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl import (
    StableDiffusionXLControlNetPipeline,
)

import cv2
from PIL import Image
import yaml

from ip_adapter import IPAdapterXL


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
    parser = argparse.ArgumentParser(description="SDXL + IP-Adapter + ControlNet(Canny) inference")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = _load_config(Path(args.config).expanduser())

    paths_cfg = cfg.get("paths", {})
    runtime_cfg = cfg.get("runtime", {})
    preprocess_cfg = cfg.get("preprocess", {})
    generate_cfg = cfg.get("generate", {})
    output_cfg = cfg.get("output", {})
    ip_adapter_cfg = cfg.get("ip_adapter", {})
    model_loading_cfg = cfg.get("model_loading", {})

    # Allow env vars as fallback when config is omitted.
    base_model_path = paths_cfg.get(
        "base_model_path",
        os.environ.get("INSTANTSTYLE_BASE_MODEL", "/root/autodl-tmp/AI_Models/stable-diffusion-xl-base-1.0/"),
    )
    image_encoder_path = paths_cfg.get(
        "image_encoder_path",
        os.environ.get("INSTANTSTYLE_IMAGE_ENCODER", "sdxl_models/image_encoder"),
    )
    ip_ckpt = paths_cfg.get("ip_ckpt", os.environ.get("INSTANTSTYLE_IP_CKPT", "sdxl_models/ip-adapter_sdxl.bin"))
    controlnet_path = paths_cfg.get("controlnet_path", "diffusers_models/controlnet-canny-sdxl-1.0")
    style_image_path = paths_cfg.get("style_image_path", "test_ref_images/blue.jpg")
    control_image_path = paths_cfg.get("control_image_path", "test_images/004.jpg")
    output_image_path = paths_cfg.get("output_image_path", "result_1.png")

    device = _get_device(runtime_cfg.get("device") or os.environ.get("INSTANTSTYLE_DEVICE"))
    torch_device = torch.device(device)
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    target_blocks = ip_adapter_cfg.get("target_blocks", ["up_blocks.0.attentions.1"])
    adain_ip = bool(ip_adapter_cfg.get("adain_ip", False))
    adain_alpha = float(ip_adapter_cfg.get("adain_alpha", 1.0))
    adain_beta = float(ip_adapter_cfg.get("adain_beta", 1.0))

    canny_low_threshold = int(preprocess_cfg.get("canny_low_threshold", 50))
    canny_high_threshold = int(preprocess_cfg.get("canny_high_threshold", 200))

    prompt = str(generate_cfg.get("prompt", "a man, masterpiece, best quality, high quality"))
    negative_prompt = str(
        generate_cfg.get(
            "negative_prompt",
            "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
        )
    )
    scale = float(generate_cfg.get("scale", 1.0))
    guidance_scale = float(generate_cfg.get("guidance_scale", 5))
    num_samples = int(generate_cfg.get("num_samples", 1))
    num_inference_steps = int(generate_cfg.get("num_inference_steps", 30))
    seed = int(generate_cfg.get("seed", 42))
    controlnet_conditioning_scale = float(generate_cfg.get("controlnet_conditioning_scale", 0.6))

    use_control_image_size = bool(output_cfg.get("use_control_image_size", True))
    configured_width = output_cfg.get("width")
    configured_height = output_cfg.get("height")

    controlnet_use_safetensors = bool(model_loading_cfg.get("controlnet_use_safetensors", False))

    base_model_dir = _require_exists(Path(base_model_path).expanduser(), "Base model directory")
    if not (base_model_dir / "model_index.json").exists():
        raise FileNotFoundError(
            "Base model directory does not look like a Diffusers SDXL model (missing model_index.json): "
            + str(base_model_dir)
        )

    image_encoder_dir = _require_exists(Path(image_encoder_path).expanduser(), "Image encoder directory")
    ip_ckpt_path = _require_exists(Path(ip_ckpt).expanduser(), "IP-Adapter checkpoint")

    controlnet_dir = _require_exists(Path(controlnet_path).expanduser(), "ControlNet directory")
    controlnet = ControlNetModel.from_pretrained(
        str(controlnet_dir),
        use_safetensors=controlnet_use_safetensors,
        torch_dtype=torch_dtype,
    ).to(torch_device)

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        str(base_model_dir),
        controlnet=controlnet,
        torch_dtype=torch_dtype,
        add_watermarker=False,
    )
    pipe.vae.enable_tiling()

    ip_model = IPAdapterXL(
        pipe,
        str(image_encoder_dir),
        str(ip_ckpt_path),
        device,
        target_blocks=target_blocks,
        adain_ip=adain_ip,
        adain_alpha=adain_alpha,
        adain_beta=adain_beta,
    )

    style_image = Image.open(_require_exists(Path(style_image_path), "Style image")).convert("RGB")
    style_image = style_image.resize((512, 512))

    control_image_file = _require_exists(Path(control_image_path), "Control image")
    input_image = cv2.imread(str(control_image_file))
    if input_image is None:
        raise FileNotFoundError(f"Control image not found or unreadable: {control_image_file}")

    detected_map = cv2.Canny(input_image, canny_low_threshold, canny_high_threshold)
    canny_map = Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB))

    control_h, control_w = input_image.shape[:2]
    if use_control_image_size:
        out_width, out_height = control_w, control_h
    else:
        if configured_width is None or configured_height is None:
            raise ValueError("When use_control_image_size is false, width and height must be set in config.")
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

    output_path = Path(output_image_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output_path)
    print(f"[info] saved: {output_path}")


if __name__ == "__main__":
    main()