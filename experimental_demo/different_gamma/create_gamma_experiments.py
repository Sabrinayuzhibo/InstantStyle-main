import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTAL_DIR = REPO_ROOT / "experimental_demo"
THIS_DIR = Path(__file__).resolve().parent

SOURCE_CONFIG = EXPERIMENTAL_DIR / "proc05_adainfalse_gamma0.6_config.yaml"
SOURCE_PAIRS = EXPERIMENTAL_DIR / "proc05_adainfalse_gamma0.6_pairs_5000.jsonl"
GAMMA_INDICATOR = THIS_DIR / "gamma_indicator" / "gamma.json"

GAMMAS = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
PROCESSOR_NAMES = [
    "up_blocks.0.attentions.1.transformer_blocks.0.attn1.processor",
    "up_blocks.0.attentions.1.transformer_blocks.1.attn1.processor",
    "up_blocks.0.attentions.1.transformer_blocks.2.attn1.processor",
    "up_blocks.0.attentions.1.transformer_blocks.3.attn1.processor",
    "up_blocks.0.attentions.1.transformer_blocks.4.attn1.processor",
]


RUNNER_TEMPLATE = '''import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experimental_demo import infer_style_controlnet


CONFIG_PATH = Path(__file__).resolve().with_name("{name}_config.yaml")
PAIRS_PATH = Path(__file__).resolve().with_name("{name}_pairs_5000.jsonl")
OUTPUT_DIR = REPO_ROOT / "experimental_demo" / "outputs" / "{name}_5000"


def main() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {{CONFIG_PATH}}")
    if not PAIRS_PATH.exists():
        raise FileNotFoundError(f"Pairs jsonl not found: {{PAIRS_PATH}}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.argv = [
        sys.argv[0],
        "--config",
        str(CONFIG_PATH),
    ]
    infer_style_controlnet.main()


if __name__ == "__main__":
    main()
'''


RUN_ALL_TEMPLATE = '''import subprocess
import sys
from pathlib import Path


GAMMAS = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
THIS_DIR = Path(__file__).resolve().parent


def main() -> None:
    for gamma in GAMMAS:
        script = THIS_DIR / f"infer_proc05_adainfalse_gamma{gamma}_5000.py"
        print(f"[gamma] running {{script.name}}", flush=True)
        subprocess.run([sys.executable, str(script)], check=True)


if __name__ == "__main__":
    main()
'''


def load_source_config() -> dict:
    if not SOURCE_CONFIG.exists():
        raise FileNotFoundError(f"Source config not found: {SOURCE_CONFIG}")
    with SOURCE_CONFIG.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_gamma_names() -> list[str]:
    if not GAMMA_INDICATOR.exists():
        return [f"proc05_adainfalse_gamma{gamma}" for gamma in GAMMAS]
    with GAMMA_INDICATOR.open("r", encoding="utf-8") as f:
        records = json.load(f)
    names = [str(item["name"]) for item in records if str(item.get("name", "")).startswith("proc05_adainfalse_gamma")]
    order = {f"proc05_adainfalse_gamma{gamma}": idx for idx, gamma in enumerate(GAMMAS)}
    names = sorted(set(names), key=lambda name: order.get(name, 999))
    return names or [f"proc05_adainfalse_gamma{gamma}" for gamma in GAMMAS]


def write_pairs(name: str, gamma: str) -> Path:
    if not SOURCE_PAIRS.exists():
        raise FileNotFoundError(f"Source pairs jsonl not found: {SOURCE_PAIRS}")

    out_path = THIS_DIR / f"{name}_pairs_5000.jsonl"
    output_dir = EXPERIMENTAL_DIR / "outputs" / f"{name}_5000"
    output_dir.mkdir(parents=True, exist_ok=True)

    with SOURCE_PAIRS.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line_no, raw in enumerate(src, start=1):
            line = raw.strip()
            if not line:
                continue
            record = json.loads(line)
            source_output = Path(str(record.get("output_image_path", f"{line_no - 1:05d}.jpg")))
            record["output_image_path"] = str(output_dir / source_output.name)
            record["adain_ip"] = False
            record["style_injection_gamma"] = float(gamma)
            record["style_injection_components"] = "key_value"
            record["style_injection_match_stats"] = False
            record["style_injection_processor_names"] = PROCESSOR_NAMES
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")

    return out_path


def write_config(source_cfg: dict, name: str, gamma: str, pairs_path: Path) -> Path:
    cfg = json.loads(json.dumps(source_cfg))
    output_dir = EXPERIMENTAL_DIR / "outputs" / f"{name}_5000"

    cfg.setdefault("paths", {})
    cfg["paths"]["pairs_jsonl"] = str(pairs_path)
    cfg["paths"]["output_dir"] = str(output_dir)
    cfg["paths"]["output_image_path"] = str(output_dir / "00000.jpg")

    cfg.setdefault("ip_adapter", {})
    cfg["ip_adapter"]["adain_ip"] = False

    cfg.setdefault("style_injection", {})
    cfg["style_injection"]["enabled"] = True
    cfg["style_injection"]["processor_names"] = PROCESSOR_NAMES
    cfg["style_injection"]["components"] = "key_value"
    cfg["style_injection"]["match_stats"] = False
    cfg["style_injection"]["gamma"] = float(gamma)
    cfg["style_injection"]["per_timestep_style_forward"] = True

    cfg.setdefault("generate", {})
    cfg["generate"]["read_prompt_from_jsonl"] = True

    out_path = THIS_DIR / f"{name}_config.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return out_path


def main() -> None:
    source_cfg = load_source_config()
    names = load_gamma_names()

    for name in names:
        gamma = name.rsplit("gamma", 1)[1]
        pairs_path = write_pairs(name, gamma)
        config_path = write_config(source_cfg, name, gamma, pairs_path)
        runner_path = THIS_DIR / f"infer_{name}_5000.py"
        runner_path.write_text(RUNNER_TEMPLATE.format(name=name), encoding="utf-8")
        print(f"[ok] {name}: {config_path.name}, {pairs_path.name}, {runner_path.name}")

    run_all_path = THIS_DIR / "run_all_gamma.py"
    run_all_path.write_text(RUN_ALL_TEMPLATE, encoding="utf-8")
    print(f"[ok] {run_all_path.name}")


if __name__ == "__main__":
    main()
