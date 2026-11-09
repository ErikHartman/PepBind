"""Shared styling and exports for the figures retained in the response."""

from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgb
import seaborn as sns


PREDICTOR_COLORS = {
    "alphafold": "#4C72B0",
    "boltz": "#DD8452",
    "both": "#8172B2",
    "v1": "#666666",
}
RESPONSE_WIDTH = 9026 / 1440


def score_tick_label(key):
    """Use short score names within the named predictor blocks."""
    name = key.lower()
    if "default" in name:
        return "Default"
    if "iptm" in name:
        return "ipTM"
    if "pae" in name:
        return "−pAE"
    if "negative_dg" in name or "ΔG" in key:
        return "−ΔG"
    if "product" in name or "r×c" in name:
        return "R×C"
    if "classification" in name or key.endswith(" C"):
        return "C"
    return "R"


def draw_score_bars(ax, estimates, groups, metric, subtitle, xlabel=False, score_fontsize=7):
    """Draw one task, keeping each predictor's equations and baselines together."""
    keys, positions, colors, blocks = [], [], [], []
    offset = 0
    for name, group_keys, color in groups:
        start = offset
        keys.extend(group_keys)
        positions.extend(offset + index for index in range(len(group_keys)))
        base_color = np.asarray(to_rgb(color))
        for key in group_keys:
            name_lower = key.lower()
            is_baseline = (
                "default" in name_lower
                or "iptm" in name_lower
                or "pae" in name_lower
                or "negative_dg" in name_lower
                or "ΔG" in key
            )
            colors.append(.55 * base_color + .45 if is_baseline else base_color)
        blocks.append((name, start, offset + len(group_keys) - 1, color))
        offset += len(group_keys) + 1.2

    # Absolute correlation changes the display only, leaving signed estimates intact.
    values = estimates.loc[keys].abs() if metric == "Pearson r" else estimates.loc[keys]
    bars = ax.bar(positions, values, width=.72, color=colors)
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7, color=".2")
    ax.set_xticks(positions, labels=[score_tick_label(key) for key in keys])
    ax.set_xlim(-.75, positions[-1] + 1.05)
    ax.set_ylim(0, 1.16)
    ax.set_yticks([0, .5, 1])
    ax.set_xlabel("Score" if xlabel else "", fontsize=8, labelpad=3)
    ax.set_ylabel("|Pearson r|" if metric == "Pearson r" else "AUROC", fontsize=8, labelpad=3)
    ax.set_title(subtitle, fontsize=9.5, loc="left", pad=28)
    ax.tick_params(axis="x", labelsize=score_fontsize, length=0, pad=3)
    ax.tick_params(axis="y", labelsize=7.5, length=2.5, pad=2)
    if metric == "AUROC":
        ax.axhline(.5, color=".7", linewidth=.7, linestyle=":", zorder=0)
    ax.grid(False)
    sns.despine(ax=ax)

    # Named color keys identify each predictor directly above its bars.
    for name, start, end, color in blocks:
        ax.plot(
            [start - .36, end + .36], [1.025, 1.025], color=color,
            linewidth=2, transform=ax.get_xaxis_transform(), clip_on=False,
        )
        ax.text(
            (start + end) / 2, 1.065, name, ha="center", va="bottom",
            fontsize=8.5, color=".2", transform=ax.get_xaxis_transform(), clip_on=False,
        )
    return bars


def save_figure(fig, path, *, bbox_inches=None):
    """Save the same page-width figure as a 300-dpi PNG and a vector PDF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    png_path, pdf_path = path.with_suffix(".png"), path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches=bbox_inches)
    fig.savefig(pdf_path, facecolor="white", bbox_inches=bbox_inches)
    return png_path, pdf_path
