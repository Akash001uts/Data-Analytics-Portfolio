"""Figures for the sentiment notebook, drawn from results.json only."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dap.common.plotting import (
    BLUE_RAMP,
    CATEGORICAL,
    SURFACE,
    TEXT,
    TEXT_SECONDARY,
    save,
    use_style,
)
from dap.sentiment.evaluate import LENGTH_LABELS

# The two VADER variants share a hue (light and dark blue), and so do the two transformers (green);
# TF-IDF gets its own colour.
COLOURS = {
    "vader_default": BLUE_RAMP[2],
    "vader_tuned": CATEGORICAL[0],
    "tfidf_logistic": CATEGORICAL[1],
    "roberta": CATEGORICAL[2],
    "finetuned": "#0b6e4b",  # a darker shade of the RoBERTa green
}


def macro_f1_chart(results: dict) -> plt.Figure:
    models = results["models"]
    names = list(COLOURS)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for y, m in enumerate(names):
        f1 = models[m]["macro_f1"]
        ax.errorbar(
            f1["estimate"],
            y,
            xerr=[[f1["estimate"] - f1["ci_low"]], [f1["ci_high"] - f1["estimate"]]],
            fmt="o",
            color=COLOURS[m],
            elinewidth=2,
            capsize=4,
            markersize=8,
            markeredgecolor="white",
        )
        ax.annotate(
            f"{f1['estimate']:.2f}",
            (f1["ci_high"], y),
            xytext=(8, -3),
            textcoords="offset points",
            fontsize=9,
            color=TEXT_SECONDARY,
        )
    ax.set_yticks(range(len(names)), [models[m]["label"] for m in names])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro-F1 on 10,000 held-out reviews (95% bootstrap interval)")
    ax.grid(axis="y", visible=False)
    ax.set_title("How well each model reads the star rating's sentiment")
    return fig


def confusion_grid(results: dict) -> plt.Figure:
    cmap = LinearSegmentedColormap.from_list("blues", [SURFACE, *BLUE_RAMP])
    fig, axes = plt.subplots(1, len(COLOURS), figsize=(3 * len(COLOURS), 3.4), sharey=True)
    short = ["neg", "neu", "pos"]
    for ax, m in zip(axes, COLOURS, strict=True):
        cm = np.array(results["models"][m]["confusion"], dtype=float)
        rows = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(rows, cmap=cmap, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    f"{rows[i, j]:.0%}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if rows[i, j] > 0.55 else TEXT,
                )
        ax.set_xticks(range(3), short)
        ax.set_yticks(range(3), short)
        ax.set_xlabel("Predicted")
        ax.grid(False)
        ax.set_title(results["models"][m]["label"], fontsize=9, fontweight="normal", loc="left")
    axes[0].set_ylabel("Actual (from stars)")
    fig.suptitle(
        "Where each model's predictions go, as a share of each actual class",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def length_chart(results: dict) -> plt.Figure:
    by_len = results["slices"]["length"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(LENGTH_LABELS))
    for m, colour in COLOURS.items():
        ax.plot(
            x,
            [by_len[b][m] for b in LENGTH_LABELS],
            marker="o",
            color=colour,
            label=results["models"][m]["label"],
        )
    ax.set_xticks(x, [f"{b}\n({by_len[b]['n']:,} reviews)" for b in LENGTH_LABELS])
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.set_title("Macro-F1 by review length")
    return fig


def make_all(results: dict, out_dir: Path) -> list[Path]:
    use_style()
    figs = {
        "01_macro_f1.png": macro_f1_chart(results),
        "01_confusion.png": confusion_grid(results),
        "01_length.png": length_chart(results),
    }
    written = []
    for name, fig in figs.items():
        written.append(save(fig, Path(out_dir) / name))
        plt.close(fig)
    return written
