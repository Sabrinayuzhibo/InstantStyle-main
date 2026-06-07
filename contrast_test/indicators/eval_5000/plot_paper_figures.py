from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "summary.json"
OUT_DIR = ROOT / "pic"
METRICS = ["fid", "art_fid", "lpips"]
METRIC_LABELS = {
    "fid": "FID ↓",
    "art_fid": "Art-FID ↓",
    "lpips": "LPIPS ↓",
}
DISPLAY_NAMES = {
    "adain_1blk": "AdaIN\n1-block",
    "adain_2blk": "AdaIN\n2-block",
    "noadain_1blk": "w/o AdaIN\n1-block\n(InstantStyle baseline)",
    "noadain_2blk": "w/o AdaIN\n2-block\n(InstantStyle baseline)",
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


def load_results() -> List[Dict]:
    with SUMMARY_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["results"]


def display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name.replace("_", "\n"))


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


def save_fig(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff_scatter(results: List[Dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    lpips_vals = np.array([r["lpips"] for r in results])
    sizes = 160 + (lpips_vals - lpips_vals.min()) / (np.ptp(lpips_vals) + 1e-8) * 520

    for r, size in zip(results, sizes):
        is_ours = r["name"] == OURS_NAME
        is_reference_ours = r["name"] == REFERENCE_OURS_NAME
        is_new_baseline = r["name"] in NEW_BASELINE_NAMES
        is_baseline = r["name"] in BASELINE_NAMES
        marker = (
            "*"
            if is_ours
            else "D"
            if is_reference_ours
            else "P"
            if is_new_baseline
            else "s"
            if is_baseline
            else "o"
        )
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
            label = f"{label}\n(Our New baseline)"
        if is_new_baseline:
            label_offsets = {
                "adain_1blk": (14, -18, "left", "top"),
                "adain_2blk": (0, -18, "center", "top"),
            }
            ox, oy, ha, va = label_offsets.get(r["name"], (0, 12, "center", "bottom"))
            ax.annotate(
                label,
                (r["fid"], r["art_fid"]),
                xytext=(ox, oy),
                textcoords="offset points",
                fontsize=8.5,
                ha=ha,
                va=va,
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.78),
            )
        elif is_ours or is_reference_ours:
            text_offsets = {
                OURS_NAME: (16, -8, "left", "center"),
                REFERENCE_OURS_NAME: (16, 16, "left", "center"),
            }
            ox, oy, ha, va = text_offsets.get(r["name"], (16, 8, "left", "center"))
            ax.annotate(
                label,
                (r["fid"], r["art_fid"]),
                xytext=(ox, oy),
                textcoords="offset points",
                fontsize=8.5,
                ha=ha,
                va=va,
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.78),
            )
        else:
            text_offsets = {
                "adain_only_no_controlnet_1blk": (12, 12, "left", "bottom"),
                "noadain_1blk": (12, 14, "left", "bottom"),
                "noadain_2blk": (12, 3, "left", "center"),
            }
            ox, oy, ha, va = text_offsets.get(r["name"], (12, 3, "left", "center"))
            ax.annotate(
                label,
                (r["fid"], r["art_fid"]),
                xytext=(ox, oy),
                textcoords="offset points",
                fontsize=8.5,
                ha=ha,
                va=va,
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.78),
            )

    ax.set_xlabel("Content distribution distance: FID ↓", fontsize=14)
    ax.set_ylabel("Style distribution distance: Art-FID ↓", fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_xlim(65, 100)
    ax.set_title("Content–Style Trade-off on 5k Evaluation Set", pad=18)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.28)
    ax.text(
        0.035,
        0.055,
        "Better",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color="#1B4332",
        bbox=dict(boxstyle="round,pad=0.35", fc="#E9F7EF", ec="#1B4332", lw=0.8),
    )
    ax.annotate("", xy=(0.02, 0.02), xytext=(0.16, 0.16), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", color="#1B4332", lw=1.4))

    handles = [
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#C81D25", markeredgecolor="black", markersize=14, label="Ours w/o AdaIN"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#F26B38", markeredgecolor="black", markersize=8, label="Ours AdaIN"),
        plt.Line2D([0], [0], marker="P", color="w", markerfacecolor="#7B2CBF", markeredgecolor="black", markersize=9, label="Our New baseline"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#2D5F8B", markeredgecolor="black", markersize=8, label="InstantStyle baselines"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#7A869A", markeredgecolor="white", markersize=9, label="Other methods"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.78), frameon=True)
    fig.tight_layout(rect=(0, 0, 0.8, 0.95))
    save_fig(fig, "figure1_content_style_tradeoff_scatter")


def plot_metric_small_multiples(results: List[Dict]) -> None:
    labels = [display_name(r["name"]) for r in results]
    x = np.arange(len(results))
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.4))
    colors = [method_color(r["name"]) for r in results]

    for ax, metric, tag in zip(axes, METRICS, ["(a)", "(b)", "(c)"]):
        vals = np.array([r[metric] for r in results])
        bars = ax.bar(x, vals, color=colors, width=0.68, edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}" if metric != "lpips" else f"{val:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)
        pad = max((vals.max() - vals.min()) * 0.35, vals.max() * 0.04)
        ax.set_ylim(max(0, vals.min() - pad), vals.max() + pad * 1.45)
        ax.set_title(f"{tag} {METRIC_LABELS[metric]}", loc="left", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right", rotation_mode="anchor")
        if metric == "art_fid":
            for tick_label, name in zip(ax.get_xticklabels(), [r["name"] for r in results]):
                if name in NEW_BASELINE_NAMES:
                    tick_label.set_text(f"{tick_label.get_text()}\n(Our New baseline)")
            ax.figure.canvas.draw_idle()
        ax.tick_params(axis="x", pad=2)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
    fig.suptitle("Quantitative Comparison Across Evaluation Metrics", y=1.02, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.98), w_pad=2.2)
    save_fig(fig, "figure2_grouped_metric_small_multiples")


def plot_rank_heatmap(results: List[Dict]) -> None:
    names = [r["name"] for r in results]
    rank_mat = []
    for r in results:
        row = []
        for metric in METRICS:
            sorted_vals = sorted((item[metric], item["name"]) for item in results)
            ranks = {name: idx + 1 for idx, (_, name) in enumerate(sorted_vals)}
            row.append(ranks[r["name"]])
        rank_mat.append(row)
    rank_mat = np.array(rank_mat)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    im = ax.imshow(rank_mat, cmap="YlGnBu_r", vmin=1, vmax=len(results), aspect="auto")
    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels([display_name(n).replace("\n", " ") for n in names])
    for i in range(rank_mat.shape[0]):
        for j in range(rank_mat.shape[1]):
            ax.text(j, i, f"#{rank_mat[i, j]}", ha="center", va="center", color="black", fontweight="bold")
    ax.set_title("Ablation Rank Heatmap (lower rank is better)", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Rank")
    fig.tight_layout()
    save_fig(fig, "figure3_ablation_rank_heatmap")


def plot_relative_improvement_vs_adain_1blk(results: List[Dict]) -> None:
    by_name = {r["name"]: r for r in results}
    baseline_name = "adain_1blk"
    baseline = by_name[baseline_name]
    ours = by_name[OURS_NAME]
    rel = [(baseline[m] - ours[m]) / baseline[m] * 100 for m in METRICS]

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    x = np.arange(len(METRICS))
    colors = ["#1B9E77" if v >= 0 else "#D95F02" for v in rel]
    bars = ax.bar(x, rel, color=colors, width=0.58, edgecolor="white", linewidth=1.0)
    ax.axhline(0, color="#222222", linewidth=0.9)
    for bar, val in zip(bars, rel):
        va = "bottom" if val >= 0 else "top"
        offset = 0.6 if val >= 0 else -0.6
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + offset,
            f"{val:+.1f}%",
            ha="center",
            va=va,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.set_ylabel("Relative reduction vs. AdaIN 1-block (%)")
    ax.set_title("Relative Improvement of proc05_adainfalse_gamma0.6 over AdaIN 1-block", fontweight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
    note = "Ours = proc05_adainfalse_gamma0.6. Positive values indicate lower error/distance."
    fig.text(
        0.5,
        0.012,
        note,
        ha="center",
        va="bottom",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="#F7F7F7", ec="#BBBBBB", lw=0.7),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_fig(fig, "figure4_relative_improvement_vs_adain_1blk")


def plot_relative_improvement(results: List[Dict]) -> None:
    by_name = {r["name"]: r for r in results}
    ours = by_name[OURS_NAME]
    rel_by_baseline = {
        baseline_name: [(by_name[baseline_name][m] - ours[m]) / by_name[baseline_name][m] * 100 for m in METRICS]
        for baseline_name in BASELINE_NAMES
    }

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    x = np.arange(len(METRICS))
    width = 0.34
    baseline_colors = ["#2D5F8B", "#4C78A8"]
    offsets = [-width / 2, width / 2]
    for baseline_name, color, offset in zip(BASELINE_NAMES, baseline_colors, offsets):
        rel = rel_by_baseline[baseline_name]
        bars = ax.bar(
            x + offset,
            rel,
            color=color,
            width=width,
            edgecolor="white",
            linewidth=1.0,
            label=display_name(baseline_name).replace("\n", " "),
        )
        for bar, val in zip(bars, rel):
            va = "bottom" if val >= 0 else "top"
            text_offset = 0.6 if val >= 0 else -0.6
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + text_offset,
                f"{val:+.1f}%",
                ha="center",
                va=va,
                fontsize=8.5,
                fontweight="bold",
            )
    ax.axhline(0, color="#222222", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.set_ylabel("Relative reduction vs. InstantStyle baselines (%)")
    ax.set_title("Relative Improvement of proc05_adainfalse_gamma0.6 over InstantStyle Baselines", fontweight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
    ax.legend(frameon=True, loc="upper right")
    note = "Ours = proc05_adainfalse_gamma0.6. Positive values indicate lower error/distance. Baselines: w/o AdaIN 1-block and w/o AdaIN 2-block."
    fig.text(
        0.5,
        0.012,
        note,
        ha="center",
        va="bottom",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="#F7F7F7", ec="#BBBBBB", lw=0.7),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_fig(fig, "figure4_relative_improvement_vs_instantstyle_baselines")


def main() -> None:
    results = load_results()
    plot_tradeoff_scatter(results)
    plot_metric_small_multiples(results)
    plot_rank_heatmap(results)
    plot_relative_improvement_vs_adain_1blk(results)
    plot_relative_improvement(results)
    print(f"Saved paper-style figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
