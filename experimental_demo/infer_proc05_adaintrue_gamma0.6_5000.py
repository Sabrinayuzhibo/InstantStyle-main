import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experimental_demo import infer_style_controlnet


CONFIG_PATH = Path(__file__).resolve().with_name("proc05_adaintrue_gamma0.6_config.yaml")
PAIRS_PATH = Path(__file__).resolve().with_name("proc05_adaintrue_gamma0.6_pairs_5000.jsonl")
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "proc05_adaintrue_gamma0.6_5000"


def main() -> None:
    # Reuse the experimental style-injection inference implementation with a
    # dedicated config. This avoids accidentally reading the 36-task ablation
    # jsonl from confi_canny_sdxl.yaml.
    if not PAIRS_PATH.exists():
        raise FileNotFoundError(f"Pairs jsonl not found: {PAIRS_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.argv = [
        sys.argv[0],
        "--config",
        str(CONFIG_PATH),
    ]
    infer_style_controlnet.main()


if __name__ == "__main__":
    main()
