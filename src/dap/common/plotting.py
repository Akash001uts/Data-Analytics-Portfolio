"""Shared chart style, so every figure in the repo looks like it belongs to the same project.

Colours come from a validated reference palette: one blue ramp for magnitude, a blue/red pair with
a grey midpoint for signed values, and at most three categorical colours on scatter plots and maps.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#f0efec"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]  # first three slots: safe together on any chart
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NEGATIVE, POSITIVE = "#2a78d6", "#e34948"  # diverging poles
MISSING = "#d9d8d4"


def use_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "text.color": TEXT,
            "legend.frameon": False,
            "lines.linewidth": 2,
        }
    )


def save(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return path
