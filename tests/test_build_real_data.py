"""Run the full build on the real data and check the result. Skipped unless the data is fetched.

The build takes under a minute, so it runs once per test session.
"""

import pytest

from dap.common import paths
from dap.common.manifest import load_manifest
from dap.health.features import ALLOWLIST, TIER_A, check_columns

_manifest = load_manifest(paths.manifest_path())
_have_data = all(s.path_in(paths.raw_dir()).exists() for s in _manifest.sources)

pytestmark = [
    pytest.mark.skipif(not _have_data, reason="raw data not fetched"),
    pytest.mark.real_data,
]


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    from dap.health.build import build

    tmp = tmp_path_factory.mktemp("build")
    out = build(out_dir=tmp, reports_dir=tmp)
    import geopandas as gpd

    return gpd.read_file(out["table"], layer="sa3").set_index("sa3_code"), out["stats"]


def test_geography(table):
    t, _ = table
    assert len(t) == 336 and t.index.is_unique
    assert t.index.str.fullmatch(r"\d{5}").all()
    assert (t.index.str[:3] == t.sa4_code).all()
    assert t.sa4_code.nunique() == 88
    assert t.crs.to_epsg() == 7844


def test_target(table):
    t, stats = table
    assert t.has_target.sum() == 327 == stats["sa3_with_target"]
    assert t.loc[t.has_target, "pph_asr"].notna().all()
    assert t.loc[~t.has_target, "pph_asr"].isna().all()
    assert {"Barkly", "East Arnhem", "West Pilbara"} <= set(stats["suppressed_sa3"])


def test_only_allowlisted_feature_columns(table):
    t, _ = table
    not_features = {
        "sa3_name",
        "sa4_code",
        "sa4_name",
        "gcc_code",
        "gcc_name",
        "state_code",
        "state_name",
        "aihw_sa3_group",
        "erp",
        "has_target",
        "geometry",
    }
    feature_cols = [c for c in t.columns if c not in not_features and not c.startswith("pph_")]
    check_columns(feature_cols)
    assert set(feature_cols) == {f.name for f in ALLOWLIST}


def test_tier_a_is_almost_complete_where_there_is_a_target(table):
    t, _ = table
    tt = t[t.has_target]
    for f in TIER_A:
        assert tt[f.name].isna().sum() <= 0.03 * len(tt), f.name


def test_derived_access_features_make_sense(table):
    t, _ = table
    shares = t[["ra_inner_regional_share", "ra_outer_regional_share", "ra_remote_share"]]
    # Two near-empty SA3s (no SEIFA population) have no shares; neither has a target.
    assert not t.loc[shares.isna().any(axis=1), "has_target"].any()
    shares = shares.dropna()
    assert ((shares >= 0) & (shares <= 1)).all().all()
    assert (shares.sum(axis=1).dropna() <= 1 + 1e-9).all()
    assert (t.km_to_public_hospital > 0).all()
    # ED-reporting hospitals are a subset of public hospitals, so they can't be closer.
    assert (t.km_to_ed_reporting_hospital >= t.km_to_public_hospital - 1e-9).all()
