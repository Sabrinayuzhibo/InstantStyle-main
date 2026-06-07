import os
import sys
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.controlnets.controlnet import ControlNetModel
from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl import (
    StableDiffusionXLControlNetPipeline,
)

import cv2
import numpy as np
from PIL import Image, ImageFilter
import yaml

# Make repo-root packages importable when running from experimental_demo.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ip_adapter import IPAdapterXL


class StyleKVInjectionController:
    def __init__(self, mode: str = "replace", gamma: float = 1.0, components: str = "value", match_stats: bool = False) -> None:
        if mode not in {"replace", "blend"}:
            raise ValueError("style_injection.mode must be 'replace' or 'blend'")
        if components not in {"key_value", "value"}:
            raise ValueError("style_injection.components must be 'key_value' or 'value'")
        self.mode = mode
        self.gamma = float(gamma)
        self.components = components
        self.match_stats = bool(match_stats)
        self.phase = "content"
        self.current_timestep: Optional[int] = None
        self.cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.capture_calls = 0
        self.inject_calls = 0
        self.total_key_delta = 0.0
        self.total_value_delta = 0.0

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def set_timestep(self, timestep: Any) -> None:
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.detach().flatten()[0].item()
        self.current_timestep = int(timestep)

    def store(self, key: torch.Tensor, value: torch.Tensor) -> None:
        if self.current_timestep is None:
            return
        self.cache[self.current_timestep] = {
            "key": key.detach().clone(),
            "value": value.detach().clone(),
        }
        self.capture_calls += 1

    def get(self) -> Optional[Dict[str, torch.Tensor]]:
        if self.current_timestep is None:
            return None
        return self.cache.get(self.current_timestep)


class StyleKVSelfAttnProcessor(nn.Module):
    def __init__(self, controller: StyleKVInjectionController) -> None:
        super().__init__()
        self.controller = controller

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if self.controller.phase == "style":
            self.controller.store(key, value)
        elif self.controller.phase == "content":
            cached = self.controller.get()
            if cached is not None and cached["key"].shape == key.shape and cached["value"].shape == value.shape:
                style_key = cached["key"].to(device=key.device, dtype=key.dtype)
                style_value = cached["value"].to(device=value.device, dtype=value.dtype)

                if self.controller.match_stats:
                    style_key = self._match_stats(style_key, key)
                    style_value = self._match_stats(style_value, value)
                style_key = torch.nan_to_num(style_key, nan=0.0, posinf=0.0, neginf=0.0)
                style_value = torch.nan_to_num(style_value, nan=0.0, posinf=0.0, neginf=0.0)

                old_key = key
                old_value = value
                gamma = self.controller.gamma
                if self.controller.components == "key_value":
                    key = (1.0 - gamma) * key + gamma * style_key
                value = (1.0 - gamma) * value + gamma * style_value
                self.controller.total_key_delta += float((key - old_key).abs().mean().detach().cpu())
                self.controller.total_value_delta += float((value - old_value).abs().mean().detach().cpu())
                self.controller.inject_calls += 1

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=0.0, neginf=0.0)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

    @staticmethod
    def _match_stats(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        reduce_dims = tuple(range(2, source.ndim))
        source_mean = source.mean(dim=reduce_dims, keepdim=True)
        source_std = source.std(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
        ref_mean = reference.mean(dim=reduce_dims, keepdim=True)
        ref_std = reference.std(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
        return (source - source_mean) / source_std * ref_std + ref_mean


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
            if "style_image_path" not in record:
                raise ValueError(f"Missing style_image_path at {jsonl_path}:{line_no}")
            if "control_image_path" not in record and "canny_image_path" not in record:
                raise ValueError(
                    f"Missing control_image_path/canny_image_path at {jsonl_path}:{line_no}"
                )
            pairs.append(record)
    if not pairs:
        raise ValueError(f"No valid records found in {jsonl_path}")
    return pairs


def main() -> None:
    args = _parse_args()
    cfg: Dict[str, Any] = {}
    if args.config:
        config_path = Path(args.config).expanduser()
        cfg = _load_config(config_path)
        config_base_dir = config_path.resolve().parent
    else:
        default_config_path = Path(__file__).resolve().with_name("confi_canny_sdxl.yaml")
        cfg = _load_config(default_config_path)
        config_base_dir = REPO_ROOT

    paths_cfg = cfg.get("paths", {})
    runtime_cfg = cfg.get("runtime", {})
    preprocess_cfg = cfg.get("preprocess", {})
    generate_cfg = cfg.get("generate", {})
    output_cfg = cfg.get("output", {})
    debug_cfg = cfg.get("debug", {})
    ip_adapter_cfg = cfg.get("ip_adapter", {})
    model_loading_cfg = cfg.get("model_loading", {})
    style_injection_cfg = cfg.get("style_injection", {})

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
    canny_image_path = paths_cfg.get("canny_image_path")
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

    style_injection_enabled = bool(style_injection_cfg.get("enabled", False))
    default_processor_names_cfg = style_injection_cfg.get("processor_names")
    if default_processor_names_cfg is None:
        default_processor_names_cfg = [
            style_injection_cfg.get(
                "processor_name",
                "up_blocks.0.attentions.1.transformer_blocks.0.attn1.processor",
            )
        ]
    if isinstance(default_processor_names_cfg, str):
        default_style_injection_processor_names = [default_processor_names_cfg]
    else:
        default_style_injection_processor_names = [str(name) for name in default_processor_names_cfg]
    style_injection_mode = str(style_injection_cfg.get("mode", "replace"))
    style_injection_gamma = float(style_injection_cfg.get("gamma", 1.0))
    style_injection_components = str(style_injection_cfg.get("components", "value"))
    style_injection_match_stats = bool(style_injection_cfg.get("match_stats", False))
    per_timestep_style_forward = bool(style_injection_cfg.get("per_timestep_style_forward", True))
    style_injection_strength = float(style_injection_cfg.get("style_forward_noise_strength", 1.0))

    base_model_dir = _require_exists(_resolve_path(str(base_model_path), config_base_dir), "Base model directory")
    if not (base_model_dir / "model_index.json").exists():
        raise FileNotFoundError(
            "Base model directory does not look like a Diffusers SDXL model (missing model_index.json): "
            + str(base_model_dir)
        )

    image_encoder_dir = _require_exists(_resolve_path(str(image_encoder_path), config_base_dir), "Image encoder directory")
    ip_ckpt_path = _require_exists(_resolve_path(str(ip_ckpt), config_base_dir), "IP-Adapter checkpoint")

    controlnet_dir = _require_exists(_resolve_path(str(controlnet_path), config_base_dir), "ControlNet directory")
    standard_bin = controlnet_dir / "diffusion_pytorch_model.bin"
    fp16_bin = controlnet_dir / "diffusion_pytorch_model.fp16.bin"
    if not standard_bin.exists() and fp16_bin.exists():
        try:
            os.symlink(fp16_bin.name, standard_bin)
            print(f"[info] created symlink for ControlNet weights: {standard_bin.name} -> {fp16_bin.name}")
        except FileExistsError:
            pass
        except OSError as exc:
            raise FileNotFoundError(
                f"ControlNet weights found as {fp16_bin.name} but diffusers expects {standard_bin.name}; "
                f"create a symlink or rename the file. Original error: {exc}"
            ) from exc
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

    style_controller: Optional[StyleKVInjectionController] = None
    if style_injection_enabled:
        style_controller = StyleKVInjectionController(
            mode=style_injection_mode,
            gamma=style_injection_gamma,
            components=style_injection_components,
            match_stats=style_injection_match_stats,
        )

    original_unet_forward = pipe.unet.forward

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

    original_attn_processors = dict(ip_model.pipe.unet.attn_processors)

    def apply_style_processors(processor_names: List[str]) -> List[str]:
        if style_controller is None:
            return []
        patched_processors = {}
        matched = []
        target_processor_set = set(processor_names)
        for proc_name, proc in original_attn_processors.items():
            if proc_name in target_processor_set:
                patched_processors[proc_name] = StyleKVSelfAttnProcessor(style_controller)
                matched.append(proc_name)
            else:
                patched_processors[proc_name] = proc
        ip_model.pipe.unet.set_attn_processor(patched_processors)
        missing = sorted(target_processor_set - set(matched))
        if missing:
            print(f"[style_injection][warn] target processors not found: {missing}")
        return matched

    default_matched = apply_style_processors(default_style_injection_processor_names) if style_controller is not None else []
    if style_controller is not None:
        print(f"[style_injection] default_matched_processors={default_matched}")

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
                "depth_image_path": canny_image_path,
                "output_image_path": str(output_image_path),
            }
        ]

    for task_idx, task in enumerate(tasks, start=1):
        style_file = _require_exists(
            _resolve_path(str(task["style_image_path"]), config_base_dir),
            f"Style image [{task_idx}]",
        )
        control_file: Optional[Path] = None
        task_control_image_path = task.get("control_image_path")
        if task_control_image_path:
            control_file = _require_exists(
                _resolve_path(str(task_control_image_path), config_base_dir),
                f"Control image [{task_idx}]",
            )

        task_depth_image_path = task.get("depth_image_path")
        if task_depth_image_path is None:
            task_depth_image_path = task.get("canny_image_path")
        depth_file: Optional[Path] = None
        if task_depth_image_path:
            depth_file = _require_exists(
                _resolve_path(str(task_depth_image_path), config_base_dir),
                f"Depth image [{task_idx}]",
            )

        if control_file is None and depth_file is None:
            raise ValueError(
                f"Task [{task_idx}] must provide control_image_path or depth_image_path"
            )

        style_image = Image.open(style_file).convert("RGB").resize((512, 512))

        if control_file is not None:
            input_image = cv2.imread(str(control_file))
            if input_image is None:
                raise FileNotFoundError(f"Control image not found or unreadable: {control_file}")
            control_h, control_w = input_image.shape[:2]
            if depth_file is not None:
                depth_src = Image.open(depth_file).convert("L")
                depth_src = depth_src.resize((control_w, control_h), Image.BICUBIC)
                depth_src = depth_src.filter(ImageFilter.GaussianBlur(radius=1.2))
                depth_arr = cv2.normalize(
                    cv2.cvtColor(np.array(depth_src), cv2.COLOR_GRAY2BGR),
                    None,
                    0,
                    255,
                    cv2.NORM_MINMAX,
                )
                canny_map = Image.fromarray(cv2.cvtColor(depth_arr.astype("uint8"), cv2.COLOR_BGR2RGB))
                print(f"[info] task {task_idx}/{len(tasks)} using generated depth from cnt={control_file} with guidance={depth_file}")
            else:
                detected_map = cv2.Canny(input_image, canny_low_threshold, canny_high_threshold)
                canny_map = Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB))
                print(f"[info] task {task_idx}/{len(tasks)} using auto canny from cnt={control_file}")
        else:
            assert depth_file is not None
            canny_map = Image.open(depth_file).convert("RGB")
            control_w, control_h = canny_map.size
            print(f"[info] task {task_idx}/{len(tasks)} using manual depth={depth_file}")
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

        print(f"[info] task {task_idx}/{len(tasks)} prompt={task_prompt}")
        print(f"[info] task {task_idx}/{len(tasks)} negative_prompt={task_negative_prompt}")

        task_adain_ip = bool(task.get("adain_ip", adain_ip))
        task_gamma = float(task.get("style_injection_gamma", style_injection_gamma))
        task_components = str(task.get("style_injection_components", style_injection_components))
        task_match_stats = bool(task.get("style_injection_match_stats", style_injection_match_stats))
        task_processor_names_cfg = task.get("style_injection_processor_names", default_style_injection_processor_names)
        if isinstance(task_processor_names_cfg, str):
            task_processor_names = [task_processor_names_cfg]
        else:
            task_processor_names = [str(name) for name in task_processor_names_cfg]

        for proc in ip_model.pipe.unet.attn_processors.values():
            if hasattr(proc, "adainIP"):
                proc.adainIP = task_adain_ip
            if hasattr(proc, "adain_call_count"):
                proc.adain_call_count = 0

        if style_controller is not None:
            style_controller.gamma = task_gamma
            style_controller.components = task_components
            style_controller.match_stats = task_match_stats
            style_controller.cache.clear()
            style_controller.capture_calls = 0
            style_controller.inject_calls = 0
            style_controller.total_key_delta = 0.0
            style_controller.total_value_delta = 0.0
            matched = apply_style_processors(task_processor_names)
            print(f"[style_injection] task={task_idx} matched_processors={len(matched)}")

        if style_controller is not None and per_timestep_style_forward:
            style_tensor = torch.from_numpy(cv2.cvtColor(cv2.imread(str(style_file)), cv2.COLOR_BGR2RGB)).to(device=device, dtype=torch_dtype)
            style_tensor = style_tensor.permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            style_tensor = torch.nn.functional.interpolate(style_tensor, size=(out_height, out_width), mode="bilinear", align_corners=False)
            with torch.no_grad():
                style_latents = pipe.vae.encode(style_tensor).latent_dist.sample()
                style_latents = style_latents * pipe.vae.config.scaling_factor

            def style_injection_unet_forward(sample, timestep, *forward_args, **forward_kwargs):
                assert style_controller is not None
                style_controller.set_timestep(timestep)
                sigma = 0.0
                if hasattr(pipe.scheduler, "sigmas"):
                    timestep_value = int(timestep.detach().flatten()[0].item()) if isinstance(timestep, torch.Tensor) else int(timestep)
                    for idx, scheduler_timestep in enumerate(pipe.scheduler.timesteps):
                        if int(scheduler_timestep.item()) == timestep_value:
                            sigma = float(pipe.scheduler.sigmas[idx].item())
                            break
                style_latents_for_step = style_latents
                if style_latents_for_step.shape[0] != sample.shape[0]:
                    repeats = (sample.shape[0] + style_latents_for_step.shape[0] - 1) // style_latents_for_step.shape[0]
                    style_latents_for_step = style_latents_for_step.repeat(repeats, 1, 1, 1)[: sample.shape[0]]
                noise = torch.randn_like(style_latents_for_step)
                style_noisy_latents = style_latents_for_step + noise * sigma
                style_noisy_latents = style_noisy_latents / ((sigma ** 2 + 1.0) ** 0.5)
                style_noisy_latents = torch.nan_to_num(style_noisy_latents, nan=0.0, posinf=0.0, neginf=0.0)

                style_kwargs = dict(forward_kwargs)
                style_kwargs.pop("down_block_additional_residuals", None)
                style_kwargs.pop("mid_block_additional_residual", None)
                style_kwargs.pop("down_intrablock_additional_residuals", None)
                if style_noisy_latents.shape[0] != sample.shape[0]:
                    repeats = sample.shape[0]
                    style_noisy_latents_for_forward = style_noisy_latents.repeat(repeats, 1, 1, 1)
                else:
                    style_noisy_latents_for_forward = style_noisy_latents
                style_controller.set_phase("style")
                with torch.no_grad():
                    original_unet_forward(style_noisy_latents_for_forward, timestep, *forward_args, **style_kwargs)
                style_controller.set_phase("content")
                style_controller.set_timestep(timestep)
                return original_unet_forward(sample, timestep, *forward_args, **forward_kwargs)

            pipe.unet.forward = style_injection_unet_forward

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

        if style_controller is not None and per_timestep_style_forward:
            pipe.unet.forward = original_unet_forward
            print(
                f"[style_injection] capture_calls={style_controller.capture_calls}, "
                f"inject_calls={style_controller.inject_calls}, cached_timesteps={len(style_controller.cache)}"
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
            if control_file is not None:
                content_stem = control_file.stem
            else:
                assert canny_file is not None
                content_stem = canny_file.stem
            base_name = f"{task_idx:04d}_sty_{style_file.stem}_cnt_{content_stem}"
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