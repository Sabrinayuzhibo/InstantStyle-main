import subprocess
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
