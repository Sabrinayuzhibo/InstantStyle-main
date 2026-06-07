import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experimental_demo import infer_style_controlnet


CONFIG_PATH = Path(__file__).resolve().with_name("proc05_adainfalse_gamma0.6_config.yaml")
PAIRS_PATH = Path(__file__).resolve().with_name("proc05_adainfalse_gamma0.6_pairs.jsonl")
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "proc05_adainfalse_gamma0.6_5000"
DEPTH_GEN_SCRIPT = Path(__file__).resolve().with_name("generate_depth_from_cnt.py")


def main() -> None:
    # Generate depth maps from cnt images before running the experiment.
    if not PAIRS_PATH.exists():
        raise FileNotFoundError(f"Pairs jsonl not found: {PAIRS_PATH}")
    if not DEPTH_GEN_SCRIPT.exists():
        raise FileNotFoundError(f"Depth generation script not found: {DEPTH_GEN_SCRIPT}")

    subprocess.run([sys.executable, str(DEPTH_GEN_SCRIPT)], cwd=str(REPO_ROOT), check=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.argv = [
        sys.argv[0],
        "--config",
        str(CONFIG_PATH),
    ]
    infer_style_controlnet.main()


if __name__ == "__main__":
    main()
