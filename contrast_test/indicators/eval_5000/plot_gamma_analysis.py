from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "summary.json"
GAMMA_SWEEP_PATH = ROOT / "outs_proc05_adainfalse__different_gamma.json"
OUT_DIR = ROOT / "pic_gamma_analysis"

METRICS = ["fid", "art_fid", "lpips"]
METRIC_LABELS = {
    "fid": "FID ↓",
    "art_fid": "Art-FID ↓",
    "lpips": "LPIPS ↓",
}

DISPLAY_NAMES = {
    "adain_1blk": "AdaIN\n1-block",
    "adain_2blk": "AdaIN\n2-block",
    "noadain_1blk": "w/o AdaIN\n1-block",
    "noadain_2blk": "w/o AdaIN\n2-block",
    "adain_only_no_controlnet_1blk": "AdaIN-only\nw/o ControlNet",
    "proc05_adaintrue_gamma0.6": "Ours\nAdaIN\nγ=0.6",
    "proc05_adainfalse_gamma0.6": "Ours\nw/o AdaIN\nγ=0.6",
}

OURS_NAME = "proc05_adainfalse_gamma0.6"
REFERENCE_OURS_NAME = "proc05_adaintrue_gamma0.6"
NEW_BASELINE_NAMES = ["adain_1blk", "adain_2blk"]
BASELINE_NAMES = ["noadain_1blk", "noadain_2blk"]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_summary() -> List[Dict]:
    with SUMMARY_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["results"]


def load_gamma_sweep() -> List[Dict]:
    with GAMMA_SWEEP_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_fig(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def method_color(name: str) -> str:
    if name == OURS_NAME:
        return "#C81D25"
    if name == REFERENCE_OURS_NAME:
        return "#F26B38"
    if name in NEW_BASELINE_NAMES:
        return "#7B2CBF" if name == "adain_1blk" else "#9D4EDD"
    if name in BASELINE_NAMES:
        return "#2D5F8B" if name == "noadain_1blk" else "#4C78A8"
    return "#7A869A"


def display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name.replace("_", "\n"))


def _gamma_value(item: Dict) -> float:
    return float(item["name"].split("gamma")[-1])


def gamma_point_style(gamma: float) -> Tuple[str, Tuple[int, int]]:
    # color, offset
    offsets = {
        0.0: ("#1F77B4", (12, -10)),
        0.2: ("#1F77B4", (10, -2)),
        0.4: ("#1F77B4", (10, -10)),
        0.8: ("#1F77B4", (-12, -22)),
        1.0: ("#1F77B4", (10, 8)),
    }
    return offsets.get(round(gamma, 1), ("#1F77B4", (10, 8)))


def plot_gamma_metric_trends(gamma_sweep: List[Dict]) -> None:
    sweep = sorted(gamma_sweep, key=_gamma_value)
    gammas = np.array([_gamma_value(item) for item in sweep])
    vals = {m: np.array([item[m] for item in sweep]) for m in METRICS}

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.6), sharex=True)
    for ax, metric in zip(axes, METRICS):
        y = vals[metric]
        ax.plot(gammas, y, color="#1F77B4", linewidth=2.0, marker="o", markersize=5.5)
        for g, yy in zip(gammas, y):
            color, (ox, oy) = gamma_point_style(g)
            if abs(g - 0.6) < 1e-8:
                continue
            ax.scatter([g], [yy], s=48, color=color, edgecolors="white", linewidths=0.8, zorder=4)
            ax.annotate(
                f"γ={g:.1f}",
                (g, yy),
                xytext=(ox, oy),
                textcoords="offset points",
                fontsize=8.3,
                color=color,
                ha="left",
                va="bottom" if oy >= 0 else "top",
            )
        if any(abs(g - 0.6) < 1e-8 for g in gammas):
            g06_idx = int(np.argmin(np.abs(gammas - 0.6)))
            ax.scatter([gammas[g06_idx]], [y[g06_idx]], s=110, color="#C81D25", marker="*", edgecolors="black", linewidths=0.8, zorder=5)
            ax.annotate(
                "γ=0.6 (ours)",
                (gammas[g06_idx], y[g06_idx]),
                xytext=(12, 12),
                textcoords="offset points",
                fontsize=8.5,
                color="#C81D25",
                fontweight="bold",
            )
        ax.set_xlabel("Gamma (γ)")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"(a) γ vs {METRIC_LABELS[metric].split(' ')[0]}", loc="left", fontweight="bold")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)
        ax.set_xticks(gammas)
    fig.suptitle("Metric Trends along Gamma Sweep", y=1.03, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97), w_pad=2.0)
    save_fig(fig, "gamma_metric_trends")


def plot_gamma_frontier(gamma_sweep: List[Dict]) -> None:
    sweep = sorted(gamma_sweep, key=_gamma_value)
    gammas = np.array([_gamma_value(item) for item in sweep])
    fid = np.array([item["fid"] for item in sweep])
    art_fid = np.array([item["art_fid"] for item in sweep])

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    ax.plot(fid, art_fid, color="#1F77B4", linewidth=2.2, alpha=0.9)
    ax.scatter(fid, art_fid, s=90, color="#1F77B4", edgecolors="white", linewidths=0.9, zorder=4)

    for item in sweep:
        g = _gamma_value(item)
        idx = int(np.argmin(np.abs(gammas - g)))
        color, (ox, oy) = gamma_point_style(g)
        if abs(g - 0.6) < 1e-8:
            color = "#C81D25"
        ax.scatter([item["fid"]], [item["art_fid"]], s=120 if abs(g - 0.6) < 1e-8 else 80, color=color, marker="*" if abs(g - 0.6) < 1e-8 else "o", edgecolors="black" if abs(g - 0.6) < 1e-8 else "white", linewidths=0.9, zorder=5)
        ax.annotate(
            f"γ={g:.1f}",
            (item["fid"], item["art_fid"]),
            xytext=(ox, oy),
            textcoords="offset points",
            fontsize=8.5,
            color=color,
            ha="left",
            va="bottom" if oy >= 0 else "top",
        )

    best_idx = int(np.argmin(art_fid))
    ax.scatter([fid[best_idx]], [art_fid[best_idx]], s=220, facecolors="none", edgecolors="#2F855A", linewidths=2.0, zorder=6)
    ax.annotate(
        f"Best balance\nγ={gammas[best_idx]:.1f}",
        (fid[best_idx], art_fid[best_idx]),
        xytext=(18, 10),
        textcoords="offset points",
        fontsize=9,
        color="#2F855A",
        fontweight="bold",
    )

    ax.set_xlabel("FID")
    ax.set_ylabel("Art-FID")
    ax.set_title("Gamma Sweep Frontier", fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)
    fig.tight_layout()
    save_fig(fig, "gamma_frontier_zoom")


def plot_pareto_front(results: List[Dict], gamma_sweep: List[Dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    lpips_vals = np.array([r["lpips"] for r in results])
    sizes = 160 + (lpips_vals - lpips_vals.min()) / (np.ptp(lpips_vals) + 1e-8) * 520

    for r, size in zip(results, sizes):
        is_ours = r["name"] == OURS_NAME
        is_reference_ours = r["name"] == REFERENCE_OURS_NAME
        is_new_baseline = r["name"] in NEW_BASELINE_NAMES
        is_baseline = r["name"] in BASELINE_NAMES
        marker = "*" if is_ours else "D" if is_reference_ours else "P" if is_new_baseline else "s" if is_baseline else "o"
        ax.scatter(
            r["fid"],
            r["art_fid"],
            s=650 if is_ours else 340 if is_reference_ours or is_new_baseline else size,
            c=method_color(r["name"]),
            marker=marker,
            edgecolors="black" if is_ours or is_reference_ours or is_new_baseline or is_baseline else "white",
            linewidths=1.35 if is_ours or is_reference_ours or is_new_baseline else 1.2 if is_baseline else 0.8,
            alpha=0.96,
            zorder=5 if is_ours or is_reference_ours or is_new_baseline or is_baseline else 3,
        )
        label = display_name(r["name"]).replace("\n", " ")
        if is_new_baseline:
            offsets = {
                "adain_1blk": (14, -18, "left", "top"),
                "adain_2blk": (0, -18, "center", "top"),
            }
            ox, oy, ha, va = offsets[r["name"]]
            ax.annotate(label, (r["fid"], r["art_fid"]), xytext=(ox, oy), textcoords="offset points", fontsize=8.5, ha=ha, va=va, zorder=6, bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.78))
        elif is_ours or is_reference_ours:
            offsets = {OURS_NAME: (16, -8, "left", "center"), REFERENCE_OURS_NAME: (16, 16, "left", "center")}
            ox, oy, ha, va = offsets[r["name"]]
            ax.annotate(label, (r["fid"], r["art_fid"]), xytext=(ox, oy), textcoords="offset points", fontsize=8.5, ha=ha, va=va, zorder=6, bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.78))
        else:
            offsets = {"adain_only_no_controlnet_1blk": (12, 12, "left", "bottom"), "noadain_1blk": (12, 14, "left", "bottom"), "noadain_2blk": (12, 3, "left", "center")}
            ox, oy, ha, va = offsets.get(r["name"], (12, 3, "left", "center"))
            ax.annotate(label, (r["fid"], r["art_fid"]), xytext=(ox, oy), textcoords="offset points", fontsize=8.5, ha=ha, va=va, zorder=6, bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.78))

    sweep = sorted(gamma_sweep, key=_gamma_value)
    gx = np.array([item["fid"] for item in sweep])
    gy = np.array([item["art_fid"] for item in sweep])
    ax.plot(gx, gy, color="#1F77B4", linewidth=2.0, alpha=0.85, zorder=2)
    ax.scatter(gx, gy, s=38, color="#1F77B4", edgecolors="white", linewidths=0.8, zorder=3)
    for item in sweep:
        g = _gamma_value(item)
        if abs(g - 0.6) < 1e-8:
            continue
        color, (ox, oy) = gamma_point_style(g)
        ax.annotate(f"γ={g:.1f}", (item["fid"], item["art_fid"]), xytext=(ox, oy), textcoords="offset points", fontsize=8.0, color=color, ha="left", va="bottom" if oy >= 0 else "top")
    g06 = next((item for item in sweep if abs(_gamma_value(item) - 0.6) < 1e-8), None)
    if g06 is not None:
        ax.scatter([g06["fid"]], [g06["art_fid"]], s=150, color="#C81D25", marker="*", edgecolors="black", linewidths=0.9, zorder=6)
        ax.annotate("γ=0.6 (ours)", (g06["fid"], g06["art_fid"]), xytext=(12, 12), textcoords="offset points", fontsize=8.8, color="#C81D25", fontweight="bold")

    ax.set_xlabel("Content distribution distance: FID ↓", fontsize=14)
    ax.set_ylabel("Style distribution distance: Art-FID ↓", fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_xlim(65, 100)
    ax.set_title("Content–Style Trade-off on 5k Evaluation Set", pad=18)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.28)
    ax.text(0.035, 0.055, "Better", transform=ax.transAxes, fontsize=12, fontweight="bold", color="#1B4332", bbox=dict(boxstyle="round,pad=0.35", fc="#E9F7EF", ec="#1B4332", lw=0.8))
    handles = [
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#C81D25", markeredgecolor="black", markersize=14, label="Ours w/o AdaIN"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#F26B38", markeredgecolor="black", markersize=8, label="Ours AdaIN"),
        plt.Line2D([0], [0], marker="P", color="w", markerfacecolor="#7B2CBF", markeredgecolor="black", markersize=9, label="Our New baseline"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#2D5F8B", markeredgecolor="black", markersize=8, label="InstantStyle baselines"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#7A869A", markeredgecolor="white", markersize=9, label="Other methods"),
        plt.Line2D([0], [0], color="#1F77B4", lw=2.0, label="γ sweep"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.78), frameon=True)
    fig.tight_layout(rect=(0, 0, 0.8, 0.95))
    save_fig(fig, "pareto_front_with_gamma_sweep")


def main() -> None:
    results = load_summary()
    gamma_sweep = load_gamma_sweep()
    plot_gamma_metric_trends(gamma_sweep)
    plot_gamma_frontier(gamma_sweep)
    plot_pareto_front(results, gamma_sweep)
    print(f"Saved gamma analysis figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
