"""Check the allowlist against the real PHIDU workbook. Skipped unless `dap health fetch` has run.

This catches PHIDU renaming a sheet or relabelling a column between releases.
"""

import pytest

from dap.common import paths
from dap.common.manifest import load_manifest
from dap.health.features import ALLOWLIST, DENY_SHEET_RE
from dap.health.phidu import resolve_features

WORKBOOK = load_manifest(paths.manifest_path()).get("phidu_sha_pha").path_in(paths.raw_dir())

pytestmark = pytest.mark.skipif(not WORKBOOK.exists(), reason="PHIDU workbook not fetched")


def test_every_phidu_feature_resolves_to_one_column():
    found = resolve_features(WORKBOOK, ALLOWLIST)
    phidu = [f for f in ALLOWLIST if f.sheet]
    assert set(found) == {f.name for f in phidu}


def test_resolved_columns_never_come_from_denylisted_sheets():
    for f in ALLOWLIST:
        if f.sheet:
            assert not DENY_SHEET_RE.match(f.sheet)
