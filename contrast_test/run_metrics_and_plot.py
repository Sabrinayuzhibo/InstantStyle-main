from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np



METRIC_KEYS = ["fid", "art_fid", "lpips"]


def _run_eval_script(gen_dir: Path, content_ref_dir: Path, style_ref_dir: Path, lpips_ref_dir: Path, device: str, num_workers: int) -> Dict:
    import subprocess
    import sys

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "eval_metrics.py"),
        "--gen_dir",
        str(gen_dir),
        "--content_ref_dir",
        str(content_ref_dir),
        "--style_ref_dir",
        str(style_ref_dir),
        "--lpips_ref_dir",
        str(lpips_ref_dir),
        "--device",
        device,
        "--num_workers",
        str(num_workers),
    ]
    print(f"[info] evaluating {gen_dir.name}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.stderr.strip():
        print(proc.stderr.strip())
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.lstrip().startswith("{"):
            return json.loads("\n".join(lines[lines.index(line):]))
    raise RuntimeError(f"Could not parse JSON from eval output for {gen_dir}\n{proc.stdout}\n{proc.stderr}")


def _save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _plot(results: List[Dict], out_path: Path) -> None:
    labels = [r["name"] for r in results]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, key in enumerate(METRIC_KEYS):
        vals = [r[key] for r in results]
        bars = ax.bar(x + (idx - 1) * width, vals, width=width, label=key)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                rotation=0,
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("Metric comparison across runs")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_single_metric(results: List[Dict], metric_key: str, out_path: Path) -> None:
    labels = [r["name"] for r in results]
    vals = [r[metric_key] for r in results]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x, vals, width=0.55, color="#2ca02c", label=metric_key)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    margin = max((max(vals) - min(vals)) * 0.25, 0.01)
    ax.set_ylim(max(0.0, min(vals) - margin), max(vals) + margin)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel(metric_key)
    ax.set_title(f"{metric_key.upper()} comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run metrics for four folders and plot comparison.")
    parser.add_argument("--content_ref_dir", required=True)
    parser.add_argument("--style_ref_dir", required=True)
    parser.add_argument("--lpips_ref_dir", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--run", action="append", nargs=2, metavar=("NAME", "GEN_DIR"), required=True)
    args = parser.parse_args()

    content_ref_dir = Path(args.content_ref_dir).expanduser().resolve()
    style_ref_dir = Path(args.style_ref_dir).expanduser().resolve()
    lpips_ref_dir = Path(args.lpips_ref_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    results: List[Dict] = []
    for name, gen in args.run:
        gen_dir = Path(gen).expanduser().resolve()
        payload = _run_eval_script(gen_dir, content_ref_dir, style_ref_dir, lpips_ref_dir, args.device, args.num_workers)
        payload["name"] = name
        results.append(payload)
        _save_json(output_root / f"{name}.json", payload)

    _save_json(output_root / "summary.json", {"results": results})
    _plot(results, output_root / "metrics_comparison.png")
    _plot_single_metric(results, "lpips", output_root / "lpips_comparison.png")


if __name__ == "__main__":
    main()
