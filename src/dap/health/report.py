"""`dap health report`: the interactive map, one self-contained HTML page.

The page uses Leaflet from cdnjs with the areas inlined as GeoJSON, and no basemap. Colours and
classes come from `figures.py`, so the interactive and static maps can't disagree, and every number
in the text comes from `reports/results.json`. The output is identical from run to run.
"""

import html
import json
from importlib import resources
from pathlib import Path
from string import Template

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from matplotlib.colors import BoundaryNorm, ListedColormap, to_hex
from shapely.geometry import MultiPolygon, mapping

from dap.common import paths
from dap.common.io import read_json
from dap.common.plotting import BLUE_RAMP, CATEGORICAL, GRID, MISSING, SURFACE, TEXT, TEXT_SECONDARY
from dap.health.build import TABLE_NAME
from dap.health.eda import quantile_classes
from dap.health.figures import CLUSTER_COLOURS, DIVERGING, RESIDUAL_BINS, RESIDUAL_LABELS
from dap.health.train import RESIDUALS_NAME, RESULTS_NAME

MAP_DIR = "map"
# Simplifying tolerances in metres (Australian Albers). Inner-city SA3s are only a few km across, so
# capitals keep more detail and regional coastlines (the Kimberley has 1,500 islands) get less.
SIMPLIFY_CITY_M = 100
SIMPLIFY_REGIONAL_M = 800
MIN_PART_KM2 = 2  # islands smaller than this are dropped (an area's largest part is always kept)
DECIMALS = 4  # about 10 m at Australian latitudes
BRANCH = "rebuild"  # switch to main at merge, with the README links
VIEWS = {
    "Australia": None,
    "Sydney": "1GSYD",
    "Melbourne": "2GMEL",
    "Brisbane": "3GBRI",
    "Adelaide": "4GADE",
    "Perth": "5GPER",
    "Hobart": "6GHOB",
    "Darwin": "7GDAR",
    "Canberra": "8ACTE",
}
CLUSTER_TEXT = {
    "High-high": "Hot spot: high rate, high-rate neighbours",
    "Low-low": "Cold spot: low rate, low-rate neighbours",
    "High-low": "High rate, low-rate neighbours",
    "Low-high": "Low rate, high-rate neighbours",
    "Not significant": "Not part of a hot or cold spot",
}
COLUMNS = ["sa3_name", "state_name", "gcc_code", "observed", "expected", "pct_vs_expected"]


def load(table_path: Path | None = None, residuals_path: Path | None = None) -> gpd.GeoDataFrame:
    """Every SA3 (suppressed ones too, so they show as grey), with the residuals joined on."""
    table_path = Path(table_path or paths.processed_dir() / TABLE_NAME)
    residuals_path = Path(residuals_path or paths.processed_dir() / RESIDUALS_NAME)
    for p, cmd in ((table_path, "build"), (residuals_path, "train")):
        if not p.exists():
            raise FileNotFoundError(f"{p} is missing; run `dap health {cmd}` first")
    table = gpd.read_file(table_path, layer="sa3").set_index("sa3_code")
    res = gpd.read_file(residuals_path, layer="sa3").set_index("sa3_code")
    cols = ["observed", "expected", "pct_vs_expected", "pph_cluster"]
    return table[["sa3_name", "state_name", "gcc_code", "geometry"]].join(pd.DataFrame(res[cols]))


def _round(geom):
    return shapely.transform(geom, lambda xy: np.round(xy, DECIMALS))


def _drop_specks(geom):
    if geom.geom_type != "MultiPolygon":
        return geom
    parts = sorted(geom.geoms, key=lambda g: g.area, reverse=True)
    keep = [parts[0]] + [g for g in parts[1:] if g.area >= MIN_PART_KM2 * 1e6]
    return MultiPolygon(keep) if len(keep) > 1 else keep[0]


def is_city(gcc_code: pd.Series) -> pd.Series:
    """Greater Capital City areas (codes like 1GSYD) and the ACT."""
    return (gcc_code.str[1] == "G") | (gcc_code == "8ACTE")


def simplify(geo: gpd.GeoDataFrame) -> gpd.GeoSeries:
    albers = geo.geometry.to_crs("EPSG:3577")
    city = is_city(geo.gcc_code)
    albers[city] = albers[city].simplify(SIMPLIFY_CITY_M)
    albers[~city] = albers[~city].simplify(SIMPLIFY_REGIONAL_M)
    out = albers.apply(_drop_specks).to_crs("EPSG:4326").apply(_round)
    if out.is_empty.any() or out.isna().any():
        raise ValueError("simplifying left some areas with no shape")
    return out


def _bounds(geom: gpd.GeoSeries) -> list[list[float]]:
    minx, miny, maxx, maxy = geom.total_bounds
    return [[round(miny, 3), round(minx, 3)], [round(maxy, 3), round(maxx, 3)]]


def layers(geo: gpd.GeoDataFrame) -> tuple[pd.DataFrame, dict]:
    """Each area's colour on each layer, and a legend (with counts) per layer."""
    has = geo.observed.notna()

    rclass = pd.cut(geo.pct_vs_expected, RESIDUAL_BINS, labels=False)
    residual = rclass.map(dict(enumerate(DIVERGING))).where(has, MISSING)

    bins = quantile_classes(geo.observed[has])
    cmap = ListedColormap(BLUE_RAMP[: len(bins) - 1])
    norm = BoundaryNorm(bins, cmap.N)
    # BoundaryNorm puts the maximum itself past the last class, so clamp it into the top one
    oclass = geo.observed.map(lambda v: np.nan if pd.isna(v) else min(int(norm(v)), cmap.N - 1))
    observed = oclass.map(lambda i: MISSING if pd.isna(i) else to_hex(cmap(int(i))))

    lisa = geo.pph_cluster.map(CLUSTER_COLOURS).where(has, MISSING)

    grey = {"colour": MISSING, "label": f"No published rate ({int((~has).sum())})"}
    rc, oc, lc = rclass.value_counts(), oclass.value_counts(), geo.pph_cluster.value_counts()
    legends = {
        "residual": {
            "title": "Observed rate compared with expected (number of SA3s)",
            "items": [
                {"colour": c, "label": f"{lab} ({int(rc.get(i, 0))})"}
                for i, (c, lab) in enumerate(zip(DIVERGING, RESIDUAL_LABELS, strict=True))
            ]
            + [grey],
        },
        "observed": {
            "title": "Admissions per 100,000, age-standardised (quantile classes, number of SA3s)",
            "items": [
                {
                    "colour": to_hex(cmap(i)),
                    "label": f"{bins[i]:,.0f} to {bins[i + 1]:,.0f} ({int(oc.get(i, 0))})",
                }
                for i in range(cmap.N)
            ]
            + [grey],
        },
        "lisa": {
            "title": "Local Moran's I clusters of the admission rate, p < 0.05 (number of SA3s)",
            "items": [
                {"colour": c, "label": f"{CLUSTER_TEXT[k]} ({int(lc.get(k, 0))})"}
                for k, c in CLUSTER_COLOURS.items()
            ]
            + [grey],
        },
    }
    colours = pd.DataFrame({"residual": residual, "observed": observed, "lisa": lisa})
    return colours, legends


def _value(x):
    return None if pd.isna(x) else round(float(x), 1)


def page_data(geo: gpd.GeoDataFrame) -> dict:
    geo = geo.sort_index()
    shapes = simplify(geo)
    colours, legends = layers(geo)
    features = []
    for code, row in geo.iterrows():
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "code": code,
                    "name": row.sa3_name,
                    "state": row.state_name,
                    "observed": _value(row.observed),
                    "expected": _value(row.expected),
                    "pct": _value(row.pct_vs_expected),
                    "cluster": CLUSTER_TEXT.get(row.pph_cluster, ""),
                    "colour": colours.loc[code].to_dict(),
                },
                "geometry": mapping(shapes[code]),
            }
        )
    views = {"Australia": _bounds(shapes)}
    for name, gcc in VIEWS.items():
        if gcc and (geo.gcc_code == gcc).any():
            views[name] = _bounds(shapes[geo.gcc_code == gcc])
    return {
        "areas": {"type": "FeatureCollection", "features": features},
        "legends": legends,
        "views": views,
        "default_view": "Australia",
        "highlight": TEXT,
    }


def _json_for_script(obj) -> str:
    text = json.dumps(obj, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
    return text.replace("</", "<\\/")  # a "</script>" inside the data would end the script tag


def render(geo: gpd.GeoDataFrame, results: dict) -> str:
    data = page_data(geo)
    res = results["residuals"]
    buttons = "\n      ".join(
        f'<button type="button" data-view="{html.escape(v)}" '
        f'aria-pressed="{"true" if v == "Australia" else "false"}">{html.escape(v)}</button>'
        for v in data["views"]
    )
    template = Template(
        resources.files("dap.health").joinpath("map_template.html").read_text(encoding="utf-8")
    )
    return template.substitute(
        surface=SURFACE,
        text=TEXT,
        text_secondary=TEXT_SECONDARY,
        grid=GRID,
        accent=CATEGORICAL[0],
        year=html.escape(results["target"]["year"]),
        model=html.escape(results["models"][res["model"]]["label"]),
        features=results["target"]["features"],
        within_10=f"{res['share_of_population_within_10pct']:.0%}",
        above_20=res["areas_over_20pct_above"],
        below_20=res["areas_over_20pct_below"],
        branch=BRANCH,
        view_buttons=buttons,
        data=_json_for_script(data),
    )


def report(out_dir: Path | None = None) -> Path:
    results = read_json(paths.reports_dir() / RESULTS_NAME)
    page = render(load(), results)
    out = Path(out_dir or paths.reports_dir() / MAP_DIR) / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8", newline="\n")
    return out
