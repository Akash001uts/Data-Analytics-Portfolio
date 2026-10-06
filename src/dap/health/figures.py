"""Figures for the modelling step (notebook 02). Each function returns a matplotlib figure."""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from matplotlib.patches import Patch

from dap.common.plotting import (
    CATEGORICAL,
    GRID,
    MISSING,
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    TEXT_SECONDARY,
    save,
    use_style,
)
from dap.health.eda import label, map_with_insets
from dap.health.models import ModelData

# Residual classes in % above or below expected, symmetric around zero.
RESIDUAL_BINS = [-np.inf, -30, -15, -5, 5, 15, 30, np.inf]
RESIDUAL_LABELS = [
    "30% or more below",
    "15 to 30% below",
    "5 to 15% below",
    "Within 5%",
    "5 to 15% above",
    "15 to 30% above",
    "30% or more above",
]


def _blend(colour: str, amount: float) -> str:
    """Mix a colour towards the neutral grey: amount 1 is the full colour, 0 is neutral."""
    a = np.array(plt.matplotlib.colors.to_rgb(colour))
    b = np.array(plt.matplotlib.colors.to_rgb(NEUTRAL))
    return to_hex(b + (a - b) * amount)


DIVERGING = [
    _blend(NEGATIVE, 1.0),
    _blend(NEGATIVE, 0.6),
    _blend(NEGATIVE, 0.3),
    NEUTRAL,
    _blend(POSITIVE, 0.3),
    _blend(POSITIVE, 0.6),
    _blend(POSITIVE, 1.0),
]
CLUSTER_COLOURS = {
    "High-high": POSITIVE,
    "Low-low": NEGATIVE,
    "High-low": _blend(POSITIVE, 0.4),
    "Low-high": _blend(NEGATIVE, 0.4),
    "Not significant": NEUTRAL,
}


def lisa_map(table: gpd.GeoDataFrame, clusters: pd.Series, title: str, what: str) -> plt.Figure:
    colours = clusters.map(CLUSTER_COLOURS)
    counts = clusters.value_counts()
    names = {
        "High-high": f"High {what}, high neighbours",
        "Low-low": f"Low {what}, low neighbours",
        "High-low": f"High {what}, low neighbours",
        "Low-high": f"Low {what}, high neighbours",
        "Not significant": "Not significant",
    }
    handles = [
        Patch(facecolor=c, label=f"{names[k]} ({counts.get(k, 0)})")
        for k, c in CLUSTER_COLOURS.items()
    ] + [Patch(facecolor=MISSING, label="No target published")]
    return map_with_insets(
        table, colours, handles, title, "Local Moran's I clusters (permutation p < 0.05)", ncol=2
    )


def residual_map(table: gpd.GeoDataFrame, pct: pd.Series, model_label: str) -> plt.Figure:
    classes = pd.cut(pct, RESIDUAL_BINS, labels=False)
    colours = classes.map(dict(enumerate(DIVERGING)))
    counts = classes.value_counts()
    handles = [
        Patch(facecolor=c, label=f"{lab} ({counts.get(i, 0)})")
        for i, (c, lab) in enumerate(zip(DIVERGING, RESIDUAL_LABELS, strict=True))
    ] + [Patch(facecolor=MISSING, label="No target published")]
    return map_with_insets(
        table,
        colours,
        handles,
        "Where admissions are higher or lower than the area's profile predicts",
        f"Observed vs expected rate ({model_label}, out-of-fold, SA4-grouped CV)",
    )


def cv_comparison(results: dict) -> plt.Figure:
    models = results["models"]
    names = list(models)[::-1]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for scheme, colour, offset, text in (
        ("random", CATEGORICAL[1], 0.14, "Random K-fold"),
        ("spatial", CATEGORICAL[0], -0.14, "Grouped by SA4"),
    ):
        mean = [models[n][scheme]["r2"]["mean"] for n in names]
        sd = [models[n][scheme]["r2"]["sd"] for n in names]
        ax.errorbar(
            mean,
            y + offset,
            xerr=sd,
            fmt="o",
            color=colour,
            ecolor=colour,
            elinewidth=1.5,
            capsize=0,
            markersize=7,
            markeredgecolor="white",
            label=text,
        )
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_yticks(y, [models[n]["label"] for n in names])
    ax.set_xlabel("R² on the rate scale, population-weighted (mean and SD over 5 repeats)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2, fontsize=8)
    ax.set_title("Random folds make every model that learns look better")
    return fig


def shap_bar(shap: pd.DataFrame, data: ModelData, top: int = 15) -> plt.Figure:
    mean_abs = shap.abs().mean().sort_values(ascending=False).head(top)[::-1]
    signs = {}
    for f in mean_abs.index:
        ok = data.X[f].notna()
        signs[f] = np.corrcoef(data.X[f][ok].rank(), shap[f][ok].rank())[0, 1]
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * top + 1))
    ax.barh(
        [label(f) for f in mean_abs.index],
        mean_abs.values,
        color=[POSITIVE if signs[f] > 0 else NEGATIVE for f in mean_abs.index],
        height=0.66,
    )
    ax.set_xlabel("Mean |SHAP value| (log rate), LightGBM fitted on all areas")
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[
            Patch(color=POSITIVE, label="Higher value, higher predicted rate"),
            Patch(color=NEGATIVE, label="Higher value, lower predicted rate"),
        ],
        loc="lower right",
        fontsize=8,
    )
    ax.set_title(f"The {top} inputs LightGBM leans on most")
    return fig


def observed_vs_expected(resid: pd.DataFrame, model_label: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    lo, hi = 1200, 11000
    ax.plot([lo, hi], [lo, hi], color=TEXT_SECONDARY, linewidth=0.8, zorder=1)
    for f, ls in ((1.3, "--"), (1 / 1.3, "--")):
        ax.plot([lo, hi], [lo * f, hi * f], color=GRID, linewidth=1, linestyle=ls, zorder=1)
    ax.scatter(
        resid.expected,
        resid.observed,
        s=18,
        color=CATEGORICAL[0],
        alpha=0.75,
        edgecolor="white",
        linewidth=0.4,
        zorder=2,
    )
    extremes = resid.reindex(resid.log_ratio.abs().sort_values().index[-6:])
    for i, r in enumerate(extremes.sort_values("observed").itertuples()):
        ax.annotate(
            r.sa3_name,
            (r.expected, r.observed),
            xytext=(
                5,
                6 if i % 2 else -9,
            ),  # alternate above and below so close labels don't collide
            textcoords="offset points",
            fontsize=7,
            color=TEXT_SECONDARY,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = [1500, 2000, 3000, 5000, 8000]
    ax.set_xticks(ticks, [f"{t:,}" for t in ticks])
    ax.set_yticks(ticks, [f"{t:,}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel(f"Expected rate ({model_label}, out-of-fold)")
    ax.set_ylabel("Observed rate per 100,000")
    ax.set_title("Observed against expected, dashed lines at ±30%")
    return fig


def make_all(table, data, lisa, resid, results, shap, out_dir: Path) -> list[Path]:
    use_style()
    best = results["best_model_spatial_cv"]
    best_label = results["models"][best]["label"]
    out_dir = Path(out_dir)
    figs = {
        "02_lisa_map.png": lisa_map(
            table, lisa.cluster, "Hot and cold spots of preventable admissions", "rate"
        ),
        "02_cv_comparison.png": cv_comparison(results),
        "02_shap.png": shap_bar(shap, data),
        "02_residual_map.png": residual_map(table, resid.pct_vs_expected, best_label),
        "02_observed_vs_expected.png": observed_vs_expected(resid, best_label),
        "02_residual_lisa_map.png": lisa_map(
            table,
            resid.resid_cluster,
            "Clusters in what the model can't explain",
            "residual",
        ),
    }
    written = []
    for name, fig in figs.items():
        written.append(save(fig, out_dir / name))
        plt.close(fig)
    return written
