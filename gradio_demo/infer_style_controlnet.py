from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
import yaml
from diffusers import StableDiffusionXLPipeline
from diffusers.models.controlnets.controlnet import ControlNetModel
from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl import (
    StableDiffusionXLControlNetPipeline,
)
from PIL import Image

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
# Make repo-root packages importable when running from gradio_demo.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ip_adapter.ip_adapter import IPAdapterXL
DEFAULT_CONFIG_PATH = REPO_ROOT / "confi_canny_sdxl.yaml"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_CANNY_DIR = SCRIPT_DIR / "canny_cache"
DEFAULT_STATE_DIR = SCRIPT_DIR / "state"
DEFAULT_STATE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineBundle:
    pipe: Any
    ip_model: IPAdapterXL
    device: str
    has_controlnet: bool


_PIPE_CACHE: Dict[Tuple[str, bool], PipelineBundle] = {}


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


def _resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else base_dir / path


def _infer_control_type(controlnet_path_value: str) -> str:
    name = Path(controlnet_path_value).name.lower()
    if "canny" in name:
        return "canny"
    if "lineart" in name:
        return "lineart"
    return name.replace(" ", "_")


def _fmt_num(v: float) -> str:
    return f"{v:g}"


def _normalize_optional_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _load_default_cfg() -> Dict[str, Any]:
    if DEFAULT_CONFIG_PATH.exists():
        return _load_config(DEFAULT_CONFIG_PATH)
    return {}


DEFAULT_CFG = _load_default_cfg()
PATHS_CFG = DEFAULT_CFG.get("paths", {})
RUNTIME_CFG = DEFAULT_CFG.get("runtime", {})
PREPROCESS_CFG = DEFAULT_CFG.get("preprocess", {})
GENERATE_CFG = DEFAULT_CFG.get("generate", {})
OUTPUT_CFG = DEFAULT_CFG.get("output", {})
IP_ADAPTER_CFG = DEFAULT_CFG.get("ip_adapter", {})
MODEL_LOADING_CFG = DEFAULT_CFG.get("model_loading", {})
CONFIG_BASE_DIR = DEFAULT_CONFIG_PATH.parent

DEFAULT_BASE_MODEL_PATH = PATHS_CFG.get("base_model_path", "/root/autodl-tmp/AI_Models/stable-diffusion-xl-base-1.0")
DEFAULT_CONTROLNET_PATH = PATHS_CFG.get("controlnet_path", "/root/autodl-tmp/InstantStyle-main/diffusers_models/controlnet-canny-sdxl-1.0")
DEFAULT_IMAGE_ENCODER_PATH = PATHS_CFG.get("image_encoder_path", "/root/autodl-tmp/InstantStyle-main/sdxl_models/image_encoder")
DEFAULT_IP_CKPT = PATHS_CFG.get("ip_ckpt", "/root/autodl-tmp/InstantStyle-main/sdxl_models/ip-adapter_sdxl.bin")
DEFAULT_OUTPUT_BASE = PATHS_CFG.get("output_dir", str(DEFAULT_OUTPUT_DIR))

DEFAULT_PROMPT = str(GENERATE_CFG.get("prompt", "best quality, high quality, high detail"))
DEFAULT_NEGATIVE_PROMPT = str(
    GENERATE_CFG.get(
        "negative_prompt",
        "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry, ugly, bad anatomy",
    )
)
DEFAULT_SCALE = float(GENERATE_CFG.get("scale", 1.0))
DEFAULT_GUIDANCE_SCALE = float(GENERATE_CFG.get("guidance_scale", 5))
DEFAULT_NUM_STEPS = int(GENERATE_CFG.get("num_inference_steps", 30))
DEFAULT_SEED = int(GENERATE_CFG.get("seed", 42))
DEFAULT_CONTROLNET_COND_SCALE = float(GENERATE_CFG.get("controlnet_conditioning_scale", 0.6))
DEFAULT_NUM_SAMPLES = int(GENERATE_CFG.get("num_samples", 1))

DEFAULT_CANNY_LOW = int(PREPROCESS_CFG.get("canny_low_threshold", 50))
DEFAULT_CANNY_HIGH = int(PREPROCESS_CFG.get("canny_high_threshold", 200))
DEFAULT_BILATERAL_D = int(PREPROCESS_CFG.get("bilateral_d", 0))
DEFAULT_BILATERAL_SIGMA_COLOR = float(PREPROCESS_CFG.get("bilateral_sigma_color", 50))
DEFAULT_BILATERAL_SIGMA_SPACE = float(PREPROCESS_CFG.get("bilateral_sigma_space", 50))
DEFAULT_CANNY_APERTURE = int(PREPROCESS_CFG.get("canny_aperture_size", 3))
DEFAULT_CANNY_L2 = bool(PREPROCESS_CFG.get("canny_l2gradient", False))

DEFAULT_USE_CONTROL_IMAGE_SIZE = bool(OUTPUT_CFG.get("use_control_image_size", True))
DEFAULT_WIDTH = OUTPUT_CFG.get("width")
DEFAULT_HEIGHT = OUTPUT_CFG.get("height")

DEFAULT_TARGET_BLOCKS = IP_ADAPTER_CFG.get("target_blocks", ["up_blocks.0.attentions.1"])
DEFAULT_ADAIN_IP = bool(IP_ADAPTER_CFG.get("adain_ip", False))
DEFAULT_ADAIN_ALPHA = float(IP_ADAPTER_CFG.get("adain_alpha", 1.0))
DEFAULT_ADAIN_BETA = float(IP_ADAPTER_CFG.get("adain_beta", 1.0))

DEFAULT_DEVICE = _get_device(RUNTIME_CFG.get("device") or os.environ.get("INSTANTSTYLE_DEVICE"))
DEFAULT_ENABLE_XFORMERS = bool(RUNTIME_CFG.get("enable_xformers", True))
DEFAULT_CONTROLNET_USE_SAFETENSORS = bool(MODEL_LOADING_CFG.get("controlnet_use_safetensors", False))
DEFAULT_BASE_MODEL_VARIANT = _normalize_optional_str(MODEL_LOADING_CFG.get("base_model_variant"))
DEFAULT_BASE_MODEL_USE_SAFETENSORS = bool(MODEL_LOADING_CFG.get("base_model_use_safetensors", True))


def _build_pipeline(base_model_path: str, controlnet_path: str, image_encoder_path: str, ip_ckpt: str, device: str, use_controlnet: bool, controlnet_use_safetensors: bool, base_model_variant: Optional[str], base_model_use_safetensors: bool, enable_xformers: bool, target_blocks: List[str], adain_ip: bool, adain_alpha: float, adain_beta: float) -> PipelineBundle:
    base_model_variant = _normalize_optional_str(base_model_variant)
    if base_model_variant:
        print(f"[info] base_model_variant={base_model_variant}")
    else:
        print("[info] base_model_variant=None")

    cache_key = (f"{base_model_path}|{controlnet_path}|{image_encoder_path}|{ip_ckpt}|{device}|variant={base_model_variant}|safetensors={base_model_use_safetensors}", use_controlnet)
    if cache_key in _PIPE_CACHE:
        return _PIPE_CACHE[cache_key]

    torch_device = torch.device(device)
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    base_model_dir = _require_exists(_resolve_path(str(base_model_path), REPO_ROOT), "SDXL base model directory")
    if not (base_model_dir / "model_index.json").exists():
        raise FileNotFoundError(
            "Base model directory does not look like a Diffusers SDXL model (missing model_index.json): "
            + str(base_model_dir)
        )
    text_encoder_dir = base_model_dir / "text_encoder"
    if text_encoder_dir.exists():
        for candidate in ["model.fp16.safetensors", "model.safetensors", "pytorch_model.bin"]:
            if (text_encoder_dir / candidate).exists():
                break
        else:
            raise FileNotFoundError(
                f"No valid text_encoder weights found in {text_encoder_dir}; expected model.fp16.safetensors or model.safetensors"
            )
    text_encoder_2_dir = base_model_dir / "text_encoder_2"
    if text_encoder_2_dir.exists():
        for candidate in ["model.fp16.safetensors", "model.safetensors", "pytorch_model.bin"]:
            if (text_encoder_2_dir / candidate).exists():
                break
        else:
            raise FileNotFoundError(
                f"No valid text_encoder_2 weights found in {text_encoder_2_dir}; expected model.fp16.safetensors or model.safetensors"
            )

    image_encoder_dir = _require_exists(_resolve_path(str(image_encoder_path), REPO_ROOT), "Image encoder directory")
    ip_ckpt_path = _require_exists(_resolve_path(str(ip_ckpt), REPO_ROOT), "IP-Adapter checkpoint")

    controlnet = None
    if use_controlnet:
        controlnet_dir = _require_exists(_resolve_path(str(controlnet_path), REPO_ROOT), "ControlNet directory")
        controlnet = ControlNetModel.from_pretrained(
            str(controlnet_dir),
            use_safetensors=controlnet_use_safetensors,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).to(torch_device)

    if controlnet is not None:
        pipe_kwargs = dict(
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            add_watermarker=False,
            low_cpu_mem_usage=True,
            use_safetensors=base_model_use_safetensors,
        )
        if base_model_variant is not None:
            pipe_kwargs["variant"] = base_model_variant
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            str(base_model_dir),
            **pipe_kwargs,
        )
    else:
        pipe_kwargs = dict(
            torch_dtype=torch_dtype,
            add_watermarker=False,
            low_cpu_mem_usage=True,
            use_safetensors=base_model_use_safetensors,
        )
        if base_model_variant is not None:
            pipe_kwargs["variant"] = base_model_variant
        pipe = StableDiffusionXLPipeline.from_pretrained(
            str(base_model_dir),
            **pipe_kwargs,
        )

    if enable_xformers:
        if device != "cuda":
            print("[warn] enable_xformers is true but device is not cuda; skip xformers")
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

    bundle = PipelineBundle(pipe=pipe, ip_model=ip_model, device=device, has_controlnet=controlnet is not None)
    _PIPE_CACHE[cache_key] = bundle
    return bundle


def _prepare_canny(control_image: Image.Image, low: int, high: int, bilateral_d: int, bilateral_sigma_color: float, bilateral_sigma_space: float, aperture_size: int, l2gradient: bool) -> Image.Image:
    input_bgr = cv2.cvtColor(np_image(control_image), cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY)
    if bilateral_d > 0:
        gray_image = cv2.bilateralFilter(
            gray_image,
            d=bilateral_d,
            sigmaColor=bilateral_sigma_color,
            sigmaSpace=bilateral_sigma_space,
        )
    detected_map = cv2.Canny(
        gray_image,
        low,
        high,
        apertureSize=aperture_size,
        L2gradient=l2gradient,
    )
    return Image.fromarray(detected_map).convert("RGB")


def np_image(image: Image.Image):
    import numpy as np

    return np.array(image.convert("RGB"))


def _save_metadata(path: Path, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def generate_image(
    control_image: Optional[Image.Image],
    style_image: Optional[Image.Image],
    use_controlnet: bool,
    use_adain: bool,
    use_canny_from_input: bool,
    prompt: str,
    negative_prompt: str,
    scale: float,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int,
    num_samples: int,
    controlnet_conditioning_scale: float,
    canny_low_threshold: int,
    canny_high_threshold: int,
    bilateral_d: int,
    bilateral_sigma_color: float,
    bilateral_sigma_space: float,
    canny_aperture_size: int,
    canny_l2gradient: bool,
    use_control_image_size: bool,
    width: Optional[int],
    height: Optional[int],
    output_dir: str,
    base_model_path: str,
    controlnet_path: str,
    image_encoder_path: str,
    ip_ckpt: str,
    device: str,
    enable_xformers: bool,
    controlnet_use_safetensors: bool,
    base_model_variant: Optional[str],
    base_model_use_safetensors: bool,
    target_blocks_text: str,
    adain_alpha: float,
    adain_beta: float,
) -> Tuple[List[Image.Image], str, str]:
    if style_image is None:
        raise gr.Error("请先上传 style 图")
    if use_controlnet and control_image is None:
        raise gr.Error("启用 ControlNet 时请先上传原图")

    target_blocks = [x.strip() for x in target_blocks_text.splitlines() if x.strip()]
    if not target_blocks:
        target_blocks = ["up_blocks.0.attentions.1"]

    bundle = _build_pipeline(
        base_model_path=base_model_path,
        controlnet_path=controlnet_path,
        image_encoder_path=image_encoder_path,
        ip_ckpt=ip_ckpt,
        device=device,
        use_controlnet=use_controlnet,
        controlnet_use_safetensors=controlnet_use_safetensors,
        base_model_variant=base_model_variant,
        base_model_use_safetensors=base_model_use_safetensors,
        enable_xformers=enable_xformers,
        target_blocks=target_blocks,
        adain_ip=use_adain,
        adain_alpha=adain_alpha,
        adain_beta=adain_beta,
    )

    pipe = bundle.pipe
    ip_model = bundle.ip_model
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    if use_controlnet:
        if use_canny_from_input:
            canny_image = _prepare_canny(
                control_image,
                canny_low_threshold,
                canny_high_threshold,
                bilateral_d,
                bilateral_sigma_color,
                bilateral_sigma_space,
                canny_aperture_size,
                canny_l2gradient,
            )
        else:
            canny_image = control_image.convert("RGB")
        out_w, out_h = canny_image.size if use_control_image_size else (int(width), int(height))
    else:
        canny_image = None
        if use_control_image_size:
            out_w, out_h = style_image.size
        else:
            if width is None or height is None:
                raise gr.Error("关闭 ControlNet 时仍需提供宽高，或者勾选“使用输入尺寸”")
            out_w, out_h = int(width), int(height)

    if not use_adain:
        # 关闭 AdaIN 时，尽量把已有 processor 的 adain 标记清掉，避免复用缓存带来意外影响
        for proc in ip_model.pipe.unet.attn_processors.values():
            if hasattr(proc, "adainIP"):
                proc.adainIP = False

    generate_kwargs = dict(
        pil_image=style_image.convert("RGB"),
        prompt=prompt,
        negative_prompt=negative_prompt,
        scale=scale,
        guidance_scale=guidance_scale,
        num_samples=num_samples,
        num_inference_steps=num_inference_steps,
        seed=seed,
        width=out_w,
        height=out_h,
    )
    if use_controlnet:
        generate_kwargs["image"] = canny_image
        generate_kwargs["controlnet_conditioning_scale"] = controlnet_conditioning_scale

    images = ip_model.generate(**generate_kwargs)

    if not isinstance(images, list):
        images = [images]

    output_base = _resolve_path(str(output_dir), REPO_ROOT)
    output_base.mkdir(parents=True, exist_ok=True)
    run_name = f"run_{'controlnet' if use_controlnet else 'no_controlnet'}_{'adain' if use_adain else 'noadain'}_{seed}"
    run_dir = output_base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    control_path = run_dir / "control_input.png"
    style_path = run_dir / "style_input.png"
    style_image.save(style_path)
    if control_image is not None:
        control_image.save(control_path)

    if canny_image is not None:
        canny_image.save(run_dir / "canny_input.png")

    for idx, img in enumerate(images, start=1):
        img.save(run_dir / f"output_{idx:02d}.png")

    used_cfg = {
        "use_controlnet": use_controlnet,
        "use_adain": use_adain,
        "use_canny_from_input": use_canny_from_input,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "scale": scale,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "num_samples": num_samples,
        "controlnet_conditioning_scale": controlnet_conditioning_scale,
        "canny_low_threshold": canny_low_threshold,
        "canny_high_threshold": canny_high_threshold,
        "bilateral_d": bilateral_d,
        "bilateral_sigma_color": bilateral_sigma_color,
        "bilateral_sigma_space": bilateral_sigma_space,
        "canny_aperture_size": canny_aperture_size,
        "canny_l2gradient": canny_l2gradient,
        "use_control_image_size": use_control_image_size,
        "width": width,
        "height": height,
        "base_model_path": base_model_path,
        "controlnet_path": controlnet_path,
        "image_encoder_path": image_encoder_path,
        "ip_ckpt": ip_ckpt,
        "device": device,
        "enable_xformers": enable_xformers,
        "controlnet_use_safetensors": controlnet_use_safetensors,
        "base_model_variant": base_model_variant,
        "base_model_use_safetensors": base_model_use_safetensors,
        "target_blocks": target_blocks,
        "adain_alpha": adain_alpha,
        "adain_beta": adain_beta,
        "output_dir": str(output_base),
        "run_dir": str(run_dir),
    }
    _save_metadata(run_dir / "used_config.yaml", used_cfg)

    status = (
        f"保存到: {run_dir}\n"
        f"ControlNet={'开启' if use_controlnet else '关闭'} | AdaIN={'开启' if use_adain else '关闭'} | "
        f"Canny={'输入图生成' if use_controlnet and use_canny_from_input else '直接使用输入图' if use_controlnet else '跳过'}"
    )
    return images, str(run_dir), status


THEME_CSS = """
:root {
  --ink: #f7efe3;
  --muted: #b9aa9a;
  --panel: rgba(28, 25, 24, 0.78);
  --panel-strong: rgba(38, 34, 31, 0.92);
  --line: rgba(255, 238, 214, 0.14);
  --gold: #f3b65f;
  --coral: #ff6f61;
  --cyan: #55d6be;
}
.gradio-container {
  max-width: 1500px !important;
  margin: 0 auto !important;
  color: var(--ink) !important;
  background:
    radial-gradient(circle at 12% 8%, rgba(255, 111, 97, 0.22), transparent 32%),
    radial-gradient(circle at 86% 12%, rgba(85, 214, 190, 0.18), transparent 30%),
    linear-gradient(135deg, #15110f 0%, #241b17 45%, #0f1716 100%) !important;
}
#instantstyle-shell {
  padding: 24px 22px 34px;
}
.hero-card {
  border: 1px solid var(--line);
  border-radius: 28px;
  padding: 28px 32px;
  margin-bottom: 22px;
  background:
    linear-gradient(135deg, rgba(243, 182, 95, 0.16), rgba(85, 214, 190, 0.08)),
    rgba(20, 17, 15, 0.72);
  box-shadow: 0 30px 90px rgba(0,0,0,0.35);
}
.hero-title {
  margin: 0;
  font-size: 42px;
  line-height: 1.05;
  letter-spacing: -0.04em;
  color: var(--ink);
}
.hero-subtitle {
  margin: 12px 0 0;
  max-width: 860px;
  color: var(--muted);
  font-size: 15px;
}
.pill-row {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 18px;
}
.pill {
  border: 1px solid rgba(243, 182, 95, 0.28);
  border-radius: 999px;
  padding: 7px 12px;
  color: #ffe6bd;
  background: rgba(243, 182, 95, 0.08);
  font-size: 12px;
}
.block, .form, .panel {
  border-color: var(--line) !important;
}
.card-panel {
  border: 1px solid var(--line) !important;
  border-radius: 24px !important;
  padding: 18px !important;
  background: var(--panel) !important;
  box-shadow: 0 18px 50px rgba(0,0,0,0.22) !important;
}
.accordion {
  border-radius: 18px !important;
  background: rgba(255,255,255,0.035) !important;
  border: 1px solid var(--line) !important;
}
button.primary, .primary {
  background: linear-gradient(135deg, var(--gold), var(--coral)) !important;
  color: #1b100b !important;
  border: 0 !important;
  font-weight: 800 !important;
  border-radius: 16px !important;
  box-shadow: 0 12px 34px rgba(255,111,97,0.25) !important;
}
textarea, input, select {
  border-radius: 14px !important;
}
#result-gallery {
  border-radius: 24px !important;
  overflow: hidden;
}
"""

with gr.Blocks(title="InstantStyle Gradio Demo", css=THEME_CSS, theme=gr.themes.Soft(primary_hue="orange", neutral_hue="stone")) as demo:
    with gr.Column(elem_id="instantstyle-shell"):
        gr.HTML(
            """
            <section class="hero-card">
              <h1 class="hero-title">InstantStyle Control Studio</h1>
              <p class="hero-subtitle">一个用于 SDXL + IP-Adapter 的风格控制前端。上传结构图与风格图，快速切换 ControlNet、AdaIN、自动 Canny 和生成参数。</p>
              <div class="pill-row">
                <span class="pill">SDXL</span>
                <span class="pill">IP-Adapter</span>
                <span class="pill">ControlNet Canny</span>
                <span class="pill">AdaIN Style Fusion</span>
              </div>
            </section>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=5, elem_classes="card-panel"):
                gr.Markdown("### 输入与核心开关")
                with gr.Row():
                    control_image = gr.Image(label="输入原图 / Control 图", type="pil", height=330)
                    style_image = gr.Image(label="Style 图", type="pil", height=330)

                with gr.Row():
                    use_controlnet = gr.Checkbox(value=True, label="使用 ControlNet")
                    use_adain = gr.Checkbox(value=DEFAULT_ADAIN_IP, label="使用 AdaIN")
                    use_canny_from_input = gr.Checkbox(value=True, label="原图自动生成 Canny")

                prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=3)
                negative_prompt = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEGATIVE_PROMPT, lines=2)

                with gr.Row():
                    seed = gr.Slider(0, 2147483647, value=DEFAULT_SEED, step=1, label="Seed")
                    num_samples = gr.Slider(1, 4, value=DEFAULT_NUM_SAMPLES, step=1, label="张数")
                with gr.Row():
                    num_steps = gr.Slider(1, 100, value=DEFAULT_NUM_STEPS, step=1, label="Steps")
                    scale = gr.Slider(0.0, 2.0, value=DEFAULT_SCALE, step=0.01, label="IP-Adapter Scale")
                    guidance_scale = gr.Slider(1.0, 20.0, value=DEFAULT_GUIDANCE_SCALE, step=0.1, label="Guidance")

            with gr.Column(scale=3, elem_classes="card-panel"):
                gr.Markdown("### 生成结果")
                run_btn = gr.Button("开始生成", variant="primary", size="lg")
                result_gallery = gr.Gallery(label="输出图像", columns=2, height=520, elem_id="result-gallery")
                run_dir = gr.Textbox(label="本次输出目录", interactive=False)
                status = gr.Textbox(label="状态", interactive=False, lines=3)

        with gr.Row():
            with gr.Column(scale=1, elem_classes="card-panel"):
                with gr.Accordion("ControlNet / Canny 参数", open=True):
                    controlnet_conditioning_scale = gr.Slider(0.0, 2.0, value=DEFAULT_CONTROLNET_COND_SCALE, step=0.01, label="ControlNet Conditioning Scale")
                    with gr.Row():
                        canny_low = gr.Slider(0, 255, value=DEFAULT_CANNY_LOW, step=1, label="Canny Low")
                        canny_high = gr.Slider(0, 255, value=DEFAULT_CANNY_HIGH, step=1, label="Canny High")
                    with gr.Row():
                        bilateral_d = gr.Slider(0, 25, value=DEFAULT_BILATERAL_D, step=1, label="Bilateral d")
                        bilateral_sigma_color = gr.Slider(0.0, 255.0, value=DEFAULT_BILATERAL_SIGMA_COLOR, step=0.1, label="sigmaColor")
                        bilateral_sigma_space = gr.Slider(0.0, 255.0, value=DEFAULT_BILATERAL_SIGMA_SPACE, step=0.1, label="sigmaSpace")
                    with gr.Row():
                        canny_aperture = gr.Dropdown([3, 5, 7], value=DEFAULT_CANNY_APERTURE, label="Aperture")
                        canny_l2 = gr.Checkbox(value=DEFAULT_CANNY_L2, label="L2 Gradient")

            with gr.Column(scale=1, elem_classes="card-panel"):
                with gr.Accordion("尺寸 / AdaIN / 模型路径", open=False):
                    use_control_image_size = gr.Checkbox(value=DEFAULT_USE_CONTROL_IMAGE_SIZE, label="使用输入图尺寸")
                    with gr.Row():
                        width = gr.Number(value=DEFAULT_WIDTH if DEFAULT_WIDTH is not None else 1024, label="Width")
                        height = gr.Number(value=DEFAULT_HEIGHT if DEFAULT_HEIGHT is not None else 1024, label="Height")
                    output_dir = gr.Textbox(value=DEFAULT_OUTPUT_BASE, label="输出目录")
                    target_blocks = gr.Textbox(value="\n".join(DEFAULT_TARGET_BLOCKS), label="target_blocks（每行一个）", lines=3)
                    with gr.Row():
                        adain_alpha = gr.Slider(0.0, 1.0, value=DEFAULT_ADAIN_ALPHA, step=0.01, label="AdaIN Alpha")
                        adain_beta = gr.Slider(0.0, 1.0, value=DEFAULT_ADAIN_BETA, step=0.01, label="AdaIN Beta")
                    with gr.Row():
                        device = gr.Dropdown(["cuda", "cpu"], value=DEFAULT_DEVICE, label="Device")
                        enable_xformers = gr.Checkbox(value=DEFAULT_ENABLE_XFORMERS, label="xformers")
                    controlnet_use_safetensors = gr.Checkbox(value=DEFAULT_CONTROLNET_USE_SAFETENSORS, label="ControlNet 使用 safetensors")
                    base_model_path = gr.Textbox(value=DEFAULT_BASE_MODEL_PATH, label="Base model path")
                    controlnet_path = gr.Textbox(value=DEFAULT_CONTROLNET_PATH, label="ControlNet path")
                    image_encoder_path = gr.Textbox(value=DEFAULT_IMAGE_ENCODER_PATH, label="Image encoder path")
                    ip_ckpt = gr.Textbox(value=DEFAULT_IP_CKPT, label="IP-Adapter ckpt")
                    base_model_variant = gr.Textbox(value=DEFAULT_BASE_MODEL_VARIANT or "", label="Base model variant（可空）")
                    base_model_use_safetensors = gr.Checkbox(value=DEFAULT_BASE_MODEL_USE_SAFETENSORS, label="Base model 使用 safetensors")

    run_btn.click(
        fn=generate_image,
        inputs=[
            control_image,
            style_image,
            use_controlnet,
            use_adain,
            use_canny_from_input,
            prompt,
            negative_prompt,
            scale,
            guidance_scale,
            num_steps,
            seed,
            num_samples,
            controlnet_conditioning_scale,
            canny_low,
            canny_high,
            bilateral_d,
            bilateral_sigma_color,
            bilateral_sigma_space,
            canny_aperture,
            canny_l2,
            use_control_image_size,
            width,
            height,
            output_dir,
            base_model_path,
            controlnet_path,
            image_encoder_path,
            ip_ckpt,
            device,
            enable_xformers,
            controlnet_use_safetensors,
            base_model_variant,
            base_model_use_safetensors,
            target_blocks,
            adain_alpha,
            adain_beta,
        ],
        outputs=[result_gallery, run_dir, status],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
