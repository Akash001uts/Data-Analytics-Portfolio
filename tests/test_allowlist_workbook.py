"""Check the allowlist against the real PHIDU workbook. Skipped unless `dap health fetch` has run.

This catches PHIDU renaming a sheet, relabelling a column or shifting rows between releases.
"""

import openpyxl
import pandas as pd
import pytest

from dap.common import paths
from dap.common.manifest import load_manifest
from dap.health.features import ALLOWLIST, DENY_SHEET_RE
from dap.health.phidu import locate, read_header, read_rows, resolve_features, to_number

_manifest = load_manifest(paths.manifest_path())
WORKBOOK = _manifest.get("phidu_sha_pha").path_in(paths.raw_dir())
CONCORDANCE = _manifest.get("phidu_sa2_pha_concordance").path_in(paths.raw_dir())
SA2_ALLOCATION = _manifest.get("abs_sa2_allocation").path_in(paths.raw_dir())

pytestmark = pytest.mark.skipif(
    not (WORKBOOK.exists() and CONCORDANCE.exists() and SA2_ALLOCATION.exists()),
    reason="raw data not fetched",
)

PHIDU_FEATURES = [f for f in ALLOWLIST if f.sheet]


@pytest.fixture(scope="module")
def workbook():
    wb = openpyxl.load_workbook(WORKBOOK, read_only=True)
    yield wb
    wb.close()


@pytest.fixture(scope="module")
def expected_codes():
    conc = pd.read_excel(CONCORDANCE, sheet_name="2021 SA2 to PHA concordance", dtype=str)
    conc = conc[conc.iloc[:, 0].str.fullmatch(r"\d{9}", na=False)]
    phas = set(conc.iloc[:, 2].str.strip())
    alloc = pd.read_excel(SA2_ALLOCATION, dtype=str)
    covered = alloc[alloc.SA2_CODE_2021.isin(set(conc.iloc[:, 0]))]
    return phas, set(covered.SA3_CODE_2021)


def test_every_phidu_feature_resolves_to_one_column():
    found = resolve_features(WORKBOOK, ALLOWLIST)
    assert set(found) == {f.name for f in PHIDU_FEATURES}


def test_no_allowlisted_sheet_is_denylisted():
    for f in PHIDU_FEATURES:
        assert not DENY_SHEET_RE.match(f.sheet)


def test_concordance_gives_the_geography_in_data_md(expected_codes):
    phas, sa3s = expected_codes
    assert (len(phas), len(sa3s)) == (1165, 336)


@pytest.mark.parametrize("sheet", sorted({f.sheet for f in PHIDU_FEATURES}))
def test_sheet_rows_hold_every_pha_then_every_sa3(workbook, expected_codes, sheet):
    phas, sa3s = expected_codes
    rows = read_rows(workbook[sheet])
    assert set(rows.pha) == phas, f"{sheet}: PHA rows before AUSTRALIA+ differ from the concordance"
    assert set(rows.pseudo) <= {f"{s}9999" for s in "12345678"}, f"{sheet}: unexpected pseudo-areas"
    assert sa3s <= set(rows.totals), (
        f"{sheet}: missing SA3 totals {sorted(sa3s - set(rows.totals))}"
    )


@pytest.mark.parametrize("feature", PHIDU_FEATURES, ids=lambda f: f.name)
def test_feature_column_is_numeric_for_almost_every_sa3(workbook, expected_codes, feature):
    _, sa3s = expected_codes
    col = locate(feature, read_header(workbook[feature.sheet]))
    rows = read_rows(workbook[feature.sheet])
    values = [to_number(rows.totals[s][col.index]) for s in sa3s]
    missing = sum(v is None for v in values)
    # Tier B's modelled estimates are unpublished for 20 SA3s, so it gets a looser limit.
    limit = 0.05 if feature.tier == "A" else 0.10
    assert missing <= limit * len(sa3s), f"{feature.name}: {missing} of {len(sa3s)} SA3s missing"
