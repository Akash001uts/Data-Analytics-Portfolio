"""The interactive map page, built from the 10-area fixture. Needs no data."""

import json
import re

import geopandas as gpd
import numpy as np
import pytest

from dap.common.plotting import MISSING
from dap.health import report as R

RESULTS = {
    "target": {"year": "2023-24", "features": 42},
    "models": {"lightgbm": {"label": "LightGBM"}},
    "residuals": {
        "model": "lightgbm",
        "share_of_population_within_10pct": 0.5,
        "areas_over_20pct_above": 3,
        "areas_over_20pct_below": 2,
    },
}


@pytest.fixture
def geo(fixtures_dir) -> gpd.GeoDataFrame:
    g = gpd.read_file(fixtures_dir / "areas.gpkg").set_index("sa3_code")
    n = len(g)
    rng = np.random.default_rng(0)
    g["state_name"] = "Synthetic"
    g["gcc_code"] = ["9GSYN"] * 5 + ["9RSYN"] * (n - 5)
    g["observed"] = rng.uniform(1500, 6000, n)
    g["expected"] = g.observed * rng.uniform(0.6, 1.4, n)
    g["pct_vs_expected"] = 100 * (g.observed / g.expected - 1)
    g["pph_cluster"] = ["High-high", "Low-low", "Not significant", "High-low", "Low-high"] * 2
    g.iloc[0, g.columns.get_indexer(["observed", "expected", "pct_vs_expected", "pph_cluster"])] = (
        np.nan
    )
    return g


def embedded(page: str) -> dict:
    m = re.search(r'<script id="map-data" type="application/json">(.*?)</script>', page, re.S)
    return json.loads(m.group(1))


def test_page_has_every_area_and_valid_json(geo):
    page = R.render(geo, RESULTS)
    data = embedded(page)
    feats = data["areas"]["features"]
    assert sorted(f["properties"]["code"] for f in feats) == sorted(geo.index)
    assert "NaN" not in page and "$" not in page.split('<script id="map-data"')[0]
    assert set(data["legends"]) == {"residual", "observed", "lisa"}
    assert list(data["views"]) == ["Australia"]  # no real capital city codes in the fixture


def test_suppressed_area_is_grey_and_null(geo):
    feats = {
        f["properties"]["code"]: f["properties"]
        for f in embedded(R.render(geo, RESULTS))["areas"]["features"]
    }
    gap = feats[geo.index[0]]
    assert gap["observed"] is None and gap["pct"] is None
    assert set(gap["colour"].values()) == {MISSING}
    assert all(MISSING not in p["colour"].values() for c, p in feats.items() if c != geo.index[0])


def test_legend_counts_add_up(geo):
    for legend in embedded(R.render(geo, RESULTS))["legends"].values():
        counts = [int(re.search(r"\((\d+)\)$", i["label"]).group(1)) for i in legend["items"]]
        assert sum(counts) == len(geo)


def test_numbers_in_the_text_come_from_results(geo):
    page = R.render(geo, RESULTS)
    assert "50% of people" in page and "3 SA3s are more than 20% above" in page
    assert "trained on 42 input features" in page and "LightGBM" in page


def test_output_is_repeatable(geo):
    assert R.render(geo, RESULTS) == R.render(geo, RESULTS)


def test_names_cannot_break_out_of_the_script_tag(geo):
    geo.iloc[1, geo.columns.get_loc("sa3_name")] = "</script><b>x"
    page = R.render(geo, RESULTS)
    assert page.count("</script>") == 3  # data, Leaflet and the page's own script
    assert embedded(page)["areas"]["features"][1]["properties"]["name"] == "</script><b>x"


def test_simplify_keeps_every_shape(geo):
    shapes = R.simplify(geo)
    assert not shapes.is_empty.any() and shapes.crs.to_epsg() == 4326


def test_city_detection():
    import pandas as pd

    assert list(R.is_city(pd.Series(["1GSYD", "1RNSW", "8ACTE", "9OTER"]))) == [
        True,
        False,
        True,
        False,
    ]


def test_missing_inputs_say_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="dap health build"):
        R.load(tmp_path / "nope.gpkg", tmp_path / "nope2.gpkg")


def test_map_footer_states_the_phidu_licence():
    from importlib import resources

    from dap.common import paths
    from dap.common.manifest import load_manifest

    licences = {
        s.licence for s in load_manifest(paths.manifest_path()).sources if s.id.startswith("phidu")
    }
    assert licences == {"CC BY-NC-SA 3.0 AU"}
    page = resources.files("dap.health").joinpath("map_template.html").read_text(encoding="utf-8")
    assert "CC BY-NC-SA 3.0 AU" in page and "by-nc-sa/3.0/au/" in page
