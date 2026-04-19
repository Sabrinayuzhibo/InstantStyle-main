import os
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def _load_pairs_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    _require_exists(jsonl_path, "Pairs jsonl file")
    pairs: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {jsonl_path}:{line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record must be an object at {jsonl_path}:{line_no}")
            if "style_image_path" not in record or "control_image_path" not in record:
                raise ValueError(
                    f"Missing style_image_path/control_image_path at {jsonl_path}:{line_no}"
                )
            pairs.append(record)
    if not pairs:
        raise ValueError(f"No valid records found in {jsonl_path}")
    return pairs


def main() -> None:
    args = _parse_args()
    cfg: Dict[str, Any] = {}
    config_base_dir = Path.cwd()
    if args.config:
        config_path = Path(args.config).expanduser()
        cfg = _load_config(config_path)
        config_base_dir = config_path.resolve().parent

    paths_cfg = cfg.get("paths", {})
    runtime_cfg = cfg.get("runtime", {})
    preprocess_cfg = cfg.get("preprocess", {})
    generate_cfg = cfg.get("generate", {})
    output_cfg = cfg.get("output", {})
    debug_cfg = cfg.get("debug", {})
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
    pairs_jsonl_path = paths_cfg.get("pairs_jsonl")
    output_dir_cfg = paths_cfg.get("output_dir")

    device = _get_device(runtime_cfg.get("device") or os.environ.get("INSTANTSTYLE_DEVICE"))
    enable_xformers = bool(runtime_cfg.get("enable_xformers", True))
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
    read_prompt_from_jsonl = bool(generate_cfg.get("read_prompt_from_jsonl", False))
    read_negative_prompt_from_jsonl = bool(generate_cfg.get("read_negative_prompt_from_jsonl", False))

    use_control_image_size = bool(output_cfg.get("use_control_image_size", True))
    configured_width = output_cfg.get("width")
    configured_height = output_cfg.get("height")

    controlnet_use_safetensors = bool(model_loading_cfg.get("controlnet_use_safetensors", False))
    base_model_variant = model_loading_cfg.get("base_model_variant")
    base_model_use_safetensors = bool(model_loading_cfg.get("base_model_use_safetensors", True))
    print_adain_stats = bool(debug_cfg.get("print_adain_stats", False))
    adain_sample_count = int(debug_cfg.get("adain_processor_sample_count", 5))

    base_model_dir = _require_exists(_resolve_path(str(base_model_path), config_base_dir), "Base model directory")
    if not (base_model_dir / "model_index.json").exists():
        raise FileNotFoundError(
            "Base model directory does not look like a Diffusers SDXL model (missing model_index.json): "
            + str(base_model_dir)
        )

    image_encoder_dir = _require_exists(_resolve_path(str(image_encoder_path), config_base_dir), "Image encoder directory")
    ip_ckpt_path = _require_exists(_resolve_path(str(ip_ckpt), config_base_dir), "IP-Adapter checkpoint")

    controlnet_dir = _require_exists(_resolve_path(str(controlnet_path), config_base_dir), "ControlNet directory")
    controlnet = ControlNetModel.from_pretrained(
        str(controlnet_dir),
        use_safetensors=controlnet_use_safetensors,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(torch_device)

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        str(base_model_dir),
        controlnet=controlnet,
        torch_dtype=torch_dtype,
        add_watermarker=False,
        low_cpu_mem_usage=True,
        variant=base_model_variant,
        use_safetensors=base_model_use_safetensors,
    )
    if enable_xformers:
        if device != "cuda":
            print("[warn] runtime.enable_xformers is true but device is not cuda; skip xformers")
        else:
            try:
                import xformers  # noqa: F401

                pipe.enable_xformers_memory_efficient_attention()
                print("[info] xformers memory efficient attention enabled")
            except Exception as exc:
                print(f"[warn] failed to enable xformers memory efficient attention: {exc}")
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

    adain_enabled_processors: List[str] = []
    if print_adain_stats:
        for proc_name, proc in ip_model.pipe.unet.attn_processors.items():
            if getattr(proc, "adainIP", False):
                adain_enabled_processors.append(proc_name)
            if hasattr(proc, "adain_call_count"):
                proc.adain_call_count = 0
        print(
            f"[debug] adain_enabled_processor_count={len(adain_enabled_processors)}"
        )
        print(
            "[debug] adain_enabled_processor_samples="
            + str(adain_enabled_processors[:adain_sample_count])
        )

    if output_dir_cfg:
        output_dir = _resolve_path(str(output_dir_cfg), config_base_dir)
    else:
        output_dir = _resolve_path(str(Path(output_image_path).parent), config_base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if pairs_jsonl_path:
        pairs_file = _resolve_path(str(pairs_jsonl_path), config_base_dir)
        tasks = _load_pairs_jsonl(pairs_file)
        print(f"[info] loaded {len(tasks)} tasks from {pairs_file}")
    else:
        tasks = [
            {
                "style_image_path": str(style_image_path),
                "control_image_path": str(control_image_path),
                "output_image_path": str(output_image_path),
            }
        ]

    for task_idx, task in enumerate(tasks, start=1):
        style_file = _require_exists(
            _resolve_path(str(task["style_image_path"]), config_base_dir),
            f"Style image [{task_idx}]",
        )
        control_file = _require_exists(
            _resolve_path(str(task["control_image_path"]), config_base_dir),
            f"Control image [{task_idx}]",
        )

        style_image = Image.open(style_file).convert("RGB").resize((512, 512))

        input_image = cv2.imread(str(control_file))
        if input_image is None:
            raise FileNotFoundError(f"Control image not found or unreadable: {control_file}")

        detected_map = cv2.Canny(input_image, canny_low_threshold, canny_high_threshold)
        canny_map = Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB))

        control_h, control_w = input_image.shape[:2]
        if use_control_image_size:
            out_width, out_height = control_w, control_h
        else:
            if configured_width is None or configured_height is None:
                raise ValueError("When use_control_image_size is false, width and height must be set in config.")
            out_width, out_height = int(configured_width), int(configured_height)

        task_prompt = str(task.get("prompt", prompt)) if read_prompt_from_jsonl else prompt
        task_negative_prompt = (
            str(task.get("negative_prompt", negative_prompt))
            if read_negative_prompt_from_jsonl
            else negative_prompt
        )

        if pairs_jsonl_path and read_prompt_from_jsonl:
            print(
                f"[info] task {task_idx}/{len(tasks)} prompt={task_prompt}"
            )
            print(
                f"[info] task {task_idx}/{len(tasks)} negative_prompt={task_negative_prompt}"
            )

        images = ip_model.generate(
            pil_image=style_image,
            prompt=task_prompt,
            negative_prompt=task_negative_prompt,
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

        custom_output = task.get("output_image_path")
        if custom_output:
            save_base = _resolve_path(str(custom_output), config_base_dir)
            save_base.parent.mkdir(parents=True, exist_ok=True)
            save_paths = [save_base]
            if len(images) > 1:
                save_paths = []
                for sample_idx in range(1, len(images) + 1):
                    save_paths.append(
                        save_base.with_name(f"{save_base.stem}_s{sample_idx:02d}{save_base.suffix}")
                    )
        else:
            base_name = f"{task_idx:04d}_sty_{style_file.stem}_cnt_{control_file.stem}"
            save_paths = [output_dir / f"{base_name}.jpg"]
            if len(images) > 1:
                save_paths = []
                for sample_idx in range(1, len(images) + 1):
                    save_paths.append(output_dir / f"{base_name}_s{sample_idx:02d}.jpg")

        for img, save_path in zip(images, save_paths):
            img.save(save_path)
            print(f"[info] saved: {save_path}")

    if print_adain_stats:
        adain_total_calls = 0
        for proc in ip_model.pipe.unet.attn_processors.values():
            adain_total_calls += int(getattr(proc, "adain_call_count", 0))
        print(f"[debug] adain_runtime_call_count={adain_total_calls}")


if __name__ == "__main__":
    main()