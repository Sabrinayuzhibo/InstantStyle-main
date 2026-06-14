import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experimental_demo import infer_style_controlnet


CONFIG_PATH = Path(__file__).resolve().with_name("proc05_adainfalse_gamma0.6_config.yaml")
PAIRS_PATH = Path(__file__).resolve().with_name("proc05_adainfalse_gamma0.6_pairs_5000.jsonl")
OUTPUT_DIR = REPO_ROOT / "experimental_demo" / "outputs" / "proc05_adainfalse_gamma0.6_5000"


def main() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
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
