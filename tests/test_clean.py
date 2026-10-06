"""Unit tests for the cleaning transforms, on small made-up tables."""

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from dap.health import clean

COND = "Potentially preventable hospitalisation (PPH) condition"
ASR = "Hospitalisations per 100,000 people (age-standardised)"


def _pph_rows(rows):
    return pd.DataFrame(
        [
            {
                "Year": year,
                "SA3 code": code,
                "SA3 group": "Inner regional",
                COND: cond,
                "Demographic group": group,
                ASR: asr,
                "Number of hospitalisations": count,
                "Estimated resident population": "50000",
            }
            for year, code, cond, group, asr, count in rows
        ]
    )


def test_tidy_pph_keeps_sa3_all_persons_and_flags_suppression():
    raw = _pph_rows(
        [
            ("2023–24", "10101", "Total PPH", "All persons", "2500", "1250"),
            ("2023–24", "10101", "Total PPH", "Males", "2600", "650"),  # dropped
            ("2023–24", "001NAT", "Total PPH", "All persons", "2617", "9"),  # dropped
            ("2023–24", "10102", "Total PPH", "All persons", "n.p.", "n.p."),
            ("2023–24", "10101", "Total separations", "All persons", "1", "1"),  # dropped
        ]
    )
    t = clean.tidy_pph(raw)
    assert list(t.sa3_code) == ["10101", "10102"]
    assert list(t.year) == ["2023-24", "2023-24"]
    assert t.suppressed.tolist() == [False, True]
    assert t.asr.iloc[0] == 2500 and np.isnan(t.asr.iloc[1])


def test_tidy_pph_refuses_unparsed_rates():
    raw = _pph_rows([("2023-24", "10101", "Total PPH", "All persons", "2,5OO", "1")])
    with pytest.raises(ValueError, match="unparsed"):
        clean.tidy_pph(raw)


def test_target_columns_are_wide_with_pre_covid_rate():
    raw = _pph_rows(
        [
            ("2023-24", "10101", "Total PPH", "All persons", "2500", "1250"),
            ("2023-24", "10101", "Total chronic", "All persons", "1000", "500"),
            ("2018-19", "10101", "Total PPH", "All persons", "2700", "1300"),
            ("2023-24", "10102", "Total PPH", "All persons", "n.p.", "n.p."),
        ]
    )
    w = clean.target_columns(clean.tidy_pph(raw))
    assert w.loc["10101", "pph_asr"] == 2500
    assert w.loc["10101", "pph_chronic_asr"] == 1000
    assert w.loc["10101", "pph_asr_2018_19"] == 2700
    assert w.has_target.to_dict() == {"10101": True, "10102": False}


def test_gp_use_picks_the_right_service_rows():
    def row(service, pct, per100, year="2023-24", code="10101", group="All persons"):
        return {
            "Year": year,
            "SA3 code": code,
            "Service ": service,  # the real sheet has a trailing space in this header
            "Demographic group": group,
            "Percentage of people who had the service (%)": pct,
            "Services per 100 people": per100,
        }

    raw = pd.DataFrame(
        [
            row("GP attendances (total)", "85", "600"),
            row("GP attendances (total)", "80", "550", group="Males"),
            row("GP attendances (total)", "84", "590", year="2022-23"),
            row("GP subtotal - After-hours", "10", "20"),
            row("Allied Health attendances (total)", "40", "100"),
        ]
    )
    g = clean.gp_use(raw)
    assert g.loc["10101"].to_dict() == {
        "gp_any_claim_pct": 85.0,
        "gp_services_per_100": 600.0,
        "gp_after_hours_per_100": 20.0,
    }


def test_remoteness_shares_are_population_weighted():
    ra = pd.DataFrame(
        {
            "SA1_CODE_2021": ["10101000001", "10101000002", "10101000003", "10102000001"],
            "RA_CODE_2021": ["10", "11", "14", "19"],  # 19 = no usual address, ignored
        }
    )
    pop = pd.DataFrame(
        {
            "sa1_code": ["10101000001", "10101000002", "10101000003", "10102000001"],
            "pop": [100, 300, 100, 50],
        }
    )
    s = clean.remoteness_shares(ra, pop)
    assert list(s.index) == ["10101"]
    assert s.loc["10101"].to_dict() == pytest.approx(
        {"ra_inner_regional_share": 0.6, "ra_outer_regional_share": 0.0, "ra_remote_share": 0.2}
    )


def test_population_weighted_mean():
    sa2 = pd.DataFrame({"sa3_code": ["A", "A", "B"], "pop": [1, 3, 0]})
    v = pd.Series([10.0, 20.0, 7.0])
    out = clean.population_weighted(sa2, v)
    assert out["A"] == pytest.approx(17.5)
    assert out["B"] == pytest.approx(7.0)  # no population: falls back to the plain mean


def test_nearest_km_in_albers():
    pts = gpd.GeoDataFrame(geometry=[Point(151.0, -34.0), Point(151.0, -33.0)], crs="EPSG:4326")
    tgt = gpd.GeoDataFrame(geometry=[Point(151.0, -34.0)], crs="EPSG:4326")
    km = clean.nearest_km(pts, tgt)
    assert km.iloc[0] == pytest.approx(0, abs=1e-6)
    assert km.iloc[1] == pytest.approx(111, rel=0.02)  # one degree of latitude


def test_hospital_points_keep_open_public_with_coordinates(fixtures_dir, tmp_path):
    ed = tmp_path / "ed.json"
    ed.write_text(
        json.dumps(
            {
                "result": [
                    {
                        "data_set_id": 1,
                        "value": 5,
                        "reporting_unit_summary": {"reporting_unit_code": "H9001"},
                    },
                    {
                        "data_set_id": 2,
                        "value": 5,
                        "reporting_unit_summary": {"reporting_unit_code": "H9002"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    h = clean.hospital_points(fixtures_dir / "reporting_units.json", ed, [1])
    # H9002 is private, H9003 closed, H9004 has no coordinates, LHN901 is not a hospital.
    assert h.code.tolist() == ["H9001"]
    assert h.reports_ed.tolist() == [True]
