"""Exploratory summaries and figures for the health project. The EDA notebook calls these."""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, to_hex
from matplotlib.patches import Patch

from dap.common.plotting import (
    BLUE_RAMP,
    CATEGORICAL,
    MISSING,
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    TEXT_SECONDARY,
)
from dap.health.features import TIER_A

# AIHW's own SA3 grouping, in order from most to least urban.
SA3_GROUP_ORDER = [
    "Major cities - higher socioeconomic areas",
    "Major cities - medium socioeconomic areas",
    "Major cities - lower socioeconomic areas",
    "Inner regional",
    "Outer regional",
    "Remote and very remote",
]
BROAD = {
    g: ("Major cities" if g.startswith("Major") else "Regional" if "regional" in g else "Remote")
    for g in SA3_GROUP_ORDER
}
CAPITALS = {"1GSYD": "Sydney", "2GMEL": "Melbourne", "3GBRI": "Brisbane", "5GPER": "Perth"}


def label(name: str) -> str:
    """Readable label for a feature name, e.g. 'gp_services_per_100' -> 'GP services per 100'."""
    words = name.replace("_pct", " (%)").replace("_asr", " (rate)").replace("_", " ")
    for old, new in [
        ("gp ", "GP "),
        ("irsd", "IRSD"),
        (" ed ", " ED "),
        ("dsp", "DSP"),
        ("ft ", "full-time "),
        ("born nes", "born in non-English-speaking country"),
        ("lf ", "labour force "),
        ("age 0 14", "age 0-14"),
        ("age 65 plus", "age 65+"),
        ("age 85 plus", "age 85+"),
        ("km to", "km to nearest"),
    ]:
        words = words.replace(old, new)
    words = words.removeprefix("ra ")
    return words[0].upper() + words[1:]


def weighted_quantile(values: pd.Series, weights: pd.Series, q: float) -> float:
    order = np.argsort(values.to_numpy())
    v, w = values.to_numpy()[order], weights.to_numpy()[order]
    cum = np.cumsum(w) - 0.5 * w
    return float(np.interp(q * w.sum(), cum, v))


def weighted_quintiles(values: pd.Series, weights: pd.Series) -> pd.Series:
    """Population-weighted quintile (1 = lowest) for each row, NaN where a value is missing."""
    ok = values.notna() & weights.notna()
    cuts = [weighted_quantile(values[ok], weights[ok], q) for q in (0.2, 0.4, 0.6, 0.8)]
    out = pd.Series(np.nan, index=values.index)
    out[ok] = np.searchsorted(cuts, values[ok], side="right") + 1
    return out


def modelling_rows(table: pd.DataFrame) -> pd.DataFrame:
    """The SA3s that will be modelled: those with a published target."""
    return table[table.has_target].copy()


def rate_by_group(t: pd.DataFrame) -> pd.DataFrame:
    """Population-weighted PPH rate per AIHW SA3 group, plus the range across SA3s."""
    rows = []
    for g in SA3_GROUP_ORDER:
        s = t[t.aihw_sa3_group == g]
        rows.append(
            {
                "group": g,
                "sa3s": len(s),
                "weighted_rate": np.average(s.pph_asr, weights=s.erp),
                "min": s.pph_asr.min(),
                "max": s.pph_asr.max(),
            }
        )
    return pd.DataFrame(rows).set_index("group")


def rate_by_irsd_quintile(t: pd.DataFrame) -> pd.DataFrame:
    """Population-weighted PPH rate per IRSD quintile (1 = most disadvantaged)."""
    q = weighted_quintiles(t.irsd_score, t.erp)
    df = t.assign(quintile=q).dropna(subset=["quintile"])
    return df.groupby("quintile").apply(
        lambda g: pd.Series(
            {"sa3s": len(g), "weighted_rate": np.average(g.pph_asr, weights=g.erp)}
        ),
        include_groups=False,
    )


def spearman(x: pd.Series, y: pd.Series) -> float:
    """Spearman correlation on the rows where both are present (Pearson on ranks, no scipy)."""
    ok = x.notna() & y.notna()
    return float(np.corrcoef(x[ok].rank(), y[ok].rank())[0, 1])


def feature_correlations(t: pd.DataFrame) -> pd.Series:
    """Spearman correlation of each Tier A feature with the PPH rate, strongest first."""
    feats = [f.name for f in TIER_A]
    corr = pd.Series({f: spearman(t[f], t.pph_asr) for f in feats})
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


def missingness(table: pd.DataFrame) -> pd.DataFrame:
    t = modelling_rows(table)
    from dap.health.features import ALLOWLIST

    rows = [
        {
            "feature": f.name,
            "tier": f.tier,
            "group": f.group,
            "missing_sa3": int(t[f.name].isna().sum()),
        }
        for f in ALLOWLIST
    ]
    df = pd.DataFrame(rows)
    return df[df.missing_sa3 > 0].sort_values("missing_sa3", ascending=False)


# ---------------------------------------------------------------------------------------------
# Figures


def _classes(values: pd.Series, k: int = 7) -> np.ndarray:
    return np.unique(np.nanquantile(values, np.linspace(0, 1, k + 1)))


def choropleth(table: gpd.GeoDataFrame, column: str = "pph_asr", title: str = "") -> plt.Figure:
    """Australia-wide map with insets for four capitals, in quantile classes of one blue ramp."""
    bins = _classes(table[column])
    cmap = ListedColormap(BLUE_RAMP[: len(bins) - 1])
    norm = BoundaryNorm(bins, cmap.N)
    colours = table[column].map(lambda v: MISSING if pd.isna(v) else to_hex(cmap(norm(v))))
    handles = [
        Patch(facecolor=cmap(i), label=f"{bins[i]:,.0f} to {bins[i + 1]:,.0f}")
        for i in range(cmap.N)
    ] + [Patch(facecolor=MISSING, label="Not published")]
    return map_with_insets(
        table,
        colours,
        handles,
        title,
        "Admissions per 100,000 (age-standardised), quantile classes",
    )


def map_with_insets(
    table: gpd.GeoDataFrame,
    colours: pd.Series,
    handles: list,
    title: str,
    legend_title: str,
    ncol: int = 4,
) -> plt.Figure:
    """Draw each area in its given colour: Australia on the left, four capitals as insets."""
    geo = table.to_crs("EPSG:3577")
    geo = geo.assign(
        geometry=geo.geometry.simplify(500), _colour=colours.reindex(geo.index).fillna(MISSING)
    )

    fig = plt.figure(figsize=(11, 9))
    main = fig.add_axes([0.0, 0.16, 0.62, 0.78])
    insets = [
        fig.add_axes([0.64 + (i % 2) * 0.18, 0.56 - (i // 2) * 0.36, 0.16, 0.3]) for i in range(4)
    ]

    def draw(ax, data):
        data.plot(ax=ax, color=data["_colour"], edgecolor="white", linewidth=0.2)
        ax.set_axis_off()

    draw(main, geo)
    main.set_title(title, loc="left")
    for ax, (code, name) in zip(insets, CAPITALS.items(), strict=True):
        draw(ax, geo[geo.gcc_code == code])
        ax.set_title(name, fontsize=9, loc="left", color=TEXT_SECONDARY, fontweight="normal")

    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        ncol=ncol,
        title=legend_title,
        title_fontsize=9,
        fontsize=9,
        alignment="left",
    )
    return fig


def group_dot_chart(by_group: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(by_group))[::-1]
    ax.hlines(
        y,
        by_group["min"],
        by_group["max"],
        color=BLUE_RAMP[1],
        linewidth=6,
        capstyle="round",
        label="Range across SA3s",
    )
    ax.scatter(
        by_group.weighted_rate,
        y,
        color=CATEGORICAL[0],
        s=60,
        zorder=3,
        edgecolor="white",
        linewidth=1.5,
        label="Population-weighted rate",
    )
    for yi, v in zip(y, by_group.weighted_rate, strict=True):
        ax.annotate(
            f"{v:,.0f}",
            (v, yi),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=TEXT_SECONDARY,
        )
    ax.set_yticks(y, [f"{g} ({n})" for g, n in zip(by_group.index, by_group.sa3s, strict=True)])
    ax.set_xlabel("Potentially preventable hospitalisations per 100,000 (age-standardised)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Rates climb with remoteness, and remote areas vary the most")
    return fig


def quintile_bar_chart(by_q: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = by_q.index.astype(int)
    ax.bar(x, by_q.weighted_rate, color=CATEGORICAL[0], width=0.62)
    for xi, v in zip(x, by_q.weighted_rate, strict=True):
        ax.annotate(
            f"{v:,.0f}",
            (xi, v),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=TEXT_SECONDARY,
        )
    ax.set_xticks(x, ["1\nmost\ndisadvantaged", "2", "3", "4", "5\nleast\ndisadvantaged"])
    ax.set_ylabel("Admissions per 100,000")
    ax.grid(axis="x", visible=False)
    ax.set_title("More disadvantaged areas have higher rates")
    return fig


def correlation_chart(corr: pd.Series, top: int = 20) -> plt.Figure:
    c = corr.head(top)[::-1]
    c.index = [label(n) for n in c.index]
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(c) + 1))
    colours = [POSITIVE if v > 0 else NEGATIVE for v in c]
    ax.barh(c.index, c.values, color=colours, height=0.66)
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman correlation with the PPH rate (2023-24)")
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[
            Patch(color=POSITIVE, label="Higher value, more admissions"),
            Patch(color=NEGATIVE, label="Higher value, fewer admissions"),
        ],
        loc="lower left",
        fontsize=8,
    )
    ax.set_title(f"The {top} inputs most related to the admission rate")
    return fig


def scatter_irsd(t: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.6))
    broad = t.aihw_sa3_group.map(BROAD)
    for colour, name in zip(CATEGORICAL, ["Major cities", "Regional", "Remote"], strict=True):
        s = t[broad == name]
        ax.scatter(
            s.irsd_score,
            s.pph_asr,
            s=np.sqrt(s.erp) / 8,
            color=colour,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.8,
            label=f"{name} ({len(s)})",
        )
    ax.set_xlabel("IRSD score (lower = more disadvantaged)")
    ax.set_ylabel("Admissions per 100,000")
    leg = ax.legend(fontsize=8, title="Dot size = population", title_fontsize=8)
    for handle in leg.legend_handles:
        handle.set_sizes([40])
    ax.set_title("Disadvantage and remoteness both matter, and they overlap")
    return fig


def national_trend(tidy_national: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    labels = {
        "pph": "All PPH",
        "pph_chronic": "Chronic",
        "pph_acute": "Acute",
        "pph_vaccine": "Vaccine-preventable",
    }
    colours = {
        "pph": "#0b0b0b",
        "pph_chronic": CATEGORICAL[0],
        "pph_acute": CATEGORICAL[1],
        "pph_vaccine": CATEGORICAL[2],
    }
    for cat, g in tidy_national.groupby("category"):
        g = g.sort_values("year")
        ax.plot(g.year, g.asr, color=colours[cat], marker="o", markersize=5)
        ax.annotate(
            labels[cat],
            (g.year.iloc[-1], g.asr.iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=TEXT_SECONDARY,
        )
    ax.axvspan("2019-20", "2021-22", color=NEUTRAL, zorder=0)
    ax.text(
        "2020-21",
        ax.get_ylim()[1] * 0.97,
        "COVID",
        ha="center",
        va="top",
        fontsize=8,
        color=TEXT_SECONDARY,
    )
    ax.set_ylabel("Per 100,000 (age-standardised)")
    ax.grid(axis="x", visible=False)
    ax.set_title("National rates dipped during COVID and have mostly recovered")
    return fig
