"""Leakage guards for the health model's feature allowlist."""

from dataclasses import replace

import pytest

from dap.health import features as F


def test_allowlist_is_valid():
    F.validate_allowlist()


def test_no_feature_from_a_hospital_use_sheet():
    for f in F.ALLOWLIST:
        if f.sheet:
            assert not F.DENY_SHEET_RE.match(f.sheet), f.name


@pytest.mark.parametrize(
    "sheet",
    [
        "Hosp_type_sex",
        "Admiss_principal_diag_persons",
        "Admissions_procedures",
        "Admissions_same_day_renal",
        "Admissions_prevent_diag_total",
        "Admissions_prevent_diag_chronic",
        "ED_total_sex",
        "ED_diagnosis",
    ],
)
def test_every_hospital_use_sheet_is_rejected(sheet):
    bad = replace(F.TIER_A[0], name="leak", sheet=sheet)
    with pytest.raises(F.LeakageError, match="hospital-use"):
        F.validate_allowlist(F.ALLOWLIST + (bad,))


def test_target_source_cannot_supply_features():
    bad = F.Feature("leak", "access", "A", F.TARGET_SOURCE)
    with pytest.raises(F.LeakageError, match="target source"):
        F.validate_allowlist(F.ALLOWLIST + (bad,))


@pytest.mark.parametrize(
    "measure",
    [
        "Number",
        "SR",
        "Sig.",
        "ASR per 100 - lower 95% C.I.",
        "RRMSE",
        "Population aged 20 years and over",
    ],
)
def test_counts_and_significance_columns_are_rejected(measure):
    bad = replace(F.TIER_A[0], name="bad", measure=measure)
    with pytest.raises(F.LeakageError, match="rate or percentage"):
        F.validate_allowlist(F.ALLOWLIST + (bad,))


def test_health_status_must_be_tier_b():
    bad = F.Feature(
        "cancer_asr",
        "health_status",
        "A",
        F.PHIDU_SOURCE,
        "Census_condition_type_total",
        "People who reported they had cancer (including remission)",
        "ASR per 100",
    )
    with pytest.raises(F.LeakageError, match="Tier B"):
        F.validate_allowlist(F.ALLOWLIST + (bad,))


def test_duplicate_column_rejected():
    dup = replace(F.TIER_A[0], name="irsd_again")
    with pytest.raises(F.LeakageError, match="same PHIDU column"):
        F.validate_allowlist(F.ALLOWLIST + (dup,))


def test_check_columns_rejects_unlisted_columns():
    F.check_columns(["irsd_score", "unemployed_pct"])
    with pytest.raises(F.LeakageError, match="pph_total"):
        F.check_columns(["irsd_score", "pph_total"])


def test_every_tier_a_group_is_populated():
    groups = {f.group for f in F.TIER_A}
    assert groups == {"socioeconomic", "demographic", "access", "gp_use", "prevention"}


def test_fixture_features_are_all_allowlisted(fixtures_dir):
    import pandas as pd

    cols = pd.read_csv(fixtures_dir / "features.csv").columns.drop("sa3_code").tolist()
    F.check_columns(cols)


def test_tier_b_is_only_health_status():
    assert F.TIER_B and all(f.group == "health_status" and f.tier == "B" for f in F.TIER_B)
    assert not set(F.feature_names(("A",))) & {f.name for f in F.TIER_B}
