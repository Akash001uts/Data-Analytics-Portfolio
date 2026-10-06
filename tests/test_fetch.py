import json
from dataclasses import replace

import pytest

from dap.common.manifest import ManifestMismatch, load_manifest
from dap.health.fetch import (
    check_expect,
    count_ed_presentations,
    count_reporting_units,
    fetch_all,
    fetch_source,
)


def test_fetch_all_downloads_and_verifies(fixtures_dir, tmp_path):
    m = load_manifest(fixtures_dir / "manifest.yaml")
    done = fetch_all(m, tmp_path)
    assert set(done) == {"fixture_target", "myhospitals_reporting_units"}
    assert done["fixture_target"].read_bytes() == (fixtures_dir / "target.csv").read_bytes()
    # A second run finds the verified file and leaves it alone.
    mtime = done["fixture_target"].stat().st_mtime_ns
    fetch_all(m, tmp_path, only=["fixture_target"])
    assert done["fixture_target"].stat().st_mtime_ns == mtime


def test_fetch_replaces_a_corrupted_local_copy(fixtures_dir, tmp_path):
    m = load_manifest(fixtures_dir / "manifest.yaml")
    path = fetch_all(m, tmp_path, only=["fixture_target"])["fixture_target"]
    path.write_text("corrupted", encoding="utf-8")
    fetch_all(m, tmp_path, only=["fixture_target"])
    assert path.read_bytes() == (fixtures_dir / "target.csv").read_bytes()


def test_unknown_source_id(fixtures_dir, tmp_path):
    with pytest.raises(KeyError, match="nope"):
        fetch_all(load_manifest(fixtures_dir / "manifest.yaml"), tmp_path, only=["nope"])


def test_reporting_unit_counts(fixtures_dir):
    doc = json.loads((fixtures_dir / "reporting_units.json").read_text(encoding="utf-8"))
    assert count_reporting_units(doc) == {
        "reporting_units": 5,
        "hospitals": 4,
        "hospitals_open_public": 2,
        "hospitals_missing_coordinates": 1,
    }


def test_ed_count_only_uses_hospitals_in_pinned_data_sets():
    def item(code, kind, ds, value):
        return {
            "data_set_id": ds,
            "value": value,
            "reporting_unit_summary": {
                "reporting_unit_code": code,
                "reporting_unit_type": {"reporting_unit_type_code": kind},
            },
        }

    doc = {
        "result": [
            item("H1", "H", 10, 5.0),
            item("H1", "H", 11, 3.0),  # same hospital, another triage category
            item("H2", "H", 10, 0.0),  # no presentations
            item("H3", "H", 99, 8.0),  # data set not pinned
            item("PG1", "PG", 10, 9.0),  # peer group, not a hospital
        ]
    }
    assert count_ed_presentations(doc, [10, 11]) == {"data_items": 5, "ed_hospitals_2023_24": 1}


def test_expect_tolerance(fixtures_dir):
    src = load_manifest(fixtures_dir / "manifest.yaml").get("myhospitals_reporting_units")
    good = {
        "reporting_units": 5,
        "hospitals": 4,
        "hospitals_open_public": 2,
        "hospitals_missing_coordinates": 1,
    }
    check_expect(src, good)
    with pytest.raises(ManifestMismatch, match="hospitals: got 6"):
        check_expect(src, {**good, "hospitals": 6})


def test_failed_download_is_quarantined(fixtures_dir, tmp_path):
    m = load_manifest(fixtures_dir / "manifest.yaml")
    src = m.get("fixture_target")
    bad = replace(src, sha256="0" * 64)
    with pytest.raises(ManifestMismatch):
        fetch_source(bad, tmp_path, fixtures_dir)
    assert not bad.path_in(tmp_path).exists()
    assert bad.path_in(tmp_path).with_name("target.csv.mismatch").exists()


def test_volatile_source_that_is_not_json_fails_clearly(fixtures_dir, tmp_path):
    src = load_manifest(fixtures_dir / "manifest.yaml").get("myhospitals_reporting_units")
    csv_instead = replace(src, url="file:target.csv")
    with pytest.raises(ManifestMismatch, match="not valid JSON"):
        fetch_source(csv_instead, tmp_path, fixtures_dir)
