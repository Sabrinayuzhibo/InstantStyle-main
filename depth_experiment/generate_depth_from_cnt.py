import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEPTH_ANYTHING_ROOT = REPO_ROOT / "Depth-Anything-V2"
DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "images" / "cnt"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "images" / "depth"
DEFAULT_CHECKPOINT = DEPTH_ANYTHING_ROOT / "checkpoints" / "depth_anything_v2_vitb.pth"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate depth maps from cnt images using Depth-Anything-V2")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--encoder", type=str, default="vitb", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--input-size", type=int, default=518)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Depth checkpoint not found: {args.checkpoint}")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_py = DEPTH_ANYTHING_ROOT / "run.py"
    if not run_py.exists():
        raise FileNotFoundError(f"Depth-Anything run.py not found: {run_py}")

    cmd = [
        sys.executable,
        str(run_py),
        "--img-path",
        str(args.input_dir),
        "--outdir",
        str(args.output_dir),
        "--encoder",
        args.encoder,
        "--input-size",
        str(args.input_size),
        "--pred-only",
        "--grayscale",
    ]

    print("Running depth generation:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(DEPTH_ANYTHING_ROOT), check=True)


if __name__ == "__main__":
    main()
