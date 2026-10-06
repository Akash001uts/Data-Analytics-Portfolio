import geopandas as gpd
import numpy as np
import pandas as pd

from dap.cli import main
from dap.common.io import read_json, write_json
from dap.common.seeds import set_seed


def test_fixture_areas_nest_in_one_sa4_each(fixtures_dir):
    areas = gpd.read_file(fixtures_dir / "areas.gpkg", layer="sa3")
    assert len(areas) == 10
    assert areas.crs.to_epsg() == 7844
    assert areas.sa3_code.is_unique
    assert (areas.sa3_code.str[:3] == areas.sa4_code).all()
    assert areas.groupby("sa3_code").sa4_code.nunique().max() == 1
    assert areas.sa4_code.nunique() == 3


def test_fixture_target_has_one_suppressed_area(fixtures_dir):
    t = pd.read_csv(fixtures_dir / "target.csv", dtype=str)
    assert (t.asr_per_100k == "n.p.").sum() == 1


def test_seed_is_repeatable():
    a = set_seed().normal(size=3)
    b = set_seed().normal(size=3)
    assert np.array_equal(a, b)


def test_write_json_is_sorted_and_round_trips(tmp_path):
    p = tmp_path / "out" / "results.json"
    write_json(p, {"b": 1, "a": [1.5, "x"]})
    assert p.read_text(encoding="utf-8").index('"a"') < p.read_text(encoding="utf-8").index('"b"')
    assert read_json(p) == {"a": [1.5, "x"], "b": 1}


def test_cli_fetch_on_fixtures(fixtures_dir, tmp_path, capsys):
    code = main(
        [
            "health",
            "fetch",
            "--manifest",
            str(fixtures_dir / "manifest.yaml"),
            "--raw-dir",
            str(tmp_path),
        ]
    )
    assert code == 0
    assert "ok  fixture_target" in capsys.readouterr().out


def test_cli_reports_unbuilt_commands(capsys):
    assert main(["health", "report"]) == 2
    assert "Phase 4" in capsys.readouterr().out


def test_cli_train_without_data_says_what_to_run(tmp_path, monkeypatch, capsys):
    (tmp_path / "pyproject.toml").write_text("")
    monkeypatch.setenv("DAP_ROOT", str(tmp_path))
    assert main(["health", "train"]) == 1
    assert "dap health build" in capsys.readouterr().err
