"""The feature allowlist for the health model.

Leakage is the main risk: every admissions, hospital-use or ED column in the PHIDU workbook overlaps
with the PPH target, so features are an explicit allowlist and `validate_allowlist` rejects anything
from those sheets, anything from the target source, and any count or significance column.

PHIDU features are identified by (sheet, block title, measure label) exactly as they appear in
the September 2026 workbook, after collapsing whitespace. docs/data.md explains each tier.
"""

import re
from dataclasses import dataclass

TARGET_SOURCE = "aihw_pph_sa3"
PHIDU_SOURCE = "phidu_sha_pha"

# Sheets whose contents are hospital admissions, PPH, procedures, dialysis or ED presentations.
DENY_SHEET_RE = re.compile(r"^(Hosp_|Admiss|ED_)")

# Value types that are never features: counts, standardised ratios, significance flags, intervals,
# model error and populations (population is a weight, not a feature).
DENY_MEASURE_RE = re.compile(
    r"^(Number|No\.|SR\b|SDR\b|Sig\.|RRMSE|Population|Total pop|Usual resident|Estimated resident"
    r"|ERP|Minimum|Maximum|Aust rank)"
    r"|lower 95|upper 95| - lo| - up"
)

GROUPS = ("socioeconomic", "demographic", "access", "gp_use", "prevention", "health_status")
TIERS = ("A", "B")


@dataclass(frozen=True)
class Feature:
    name: str
    group: str
    tier: str
    source: str
    sheet: str | None = None
    block: str | None = None
    measure: str | None = None
    year: str = ""
    note: str = ""


def _p(name, group, sheet, block, measure, year, tier="A", note=""):
    return Feature(name, group, tier, PHIDU_SOURCE, sheet, block, measure, year, note)


TIER_A: tuple[Feature, ...] = (
    # Socioeconomic
    _p(
        "irsd_score",
        "socioeconomic",
        "IRSD",
        "SEIFA Index of Relative Socio-economic Disadvantage",
        "Index score (based on Australian score = 1000)",
        "2021",
    ),
    _p(
        "unemployed_pct",
        "socioeconomic",
        "Labour_force",
        "Unemployment",
        "% unemployed",
        "June 2025",
    ),
    _p(
        "lf_participation_pct",
        "socioeconomic",
        "Labour_force",
        "Labour force participation",
        "% labour force participation",
        "June 2025",
    ),
    _p(
        "dsp_pct",
        "socioeconomic",
        "Income_support",
        "Disability Support Pensioners +++",
        "% disability support pensioners",
        "June 2025",
    ),
    _p(
        "unemployment_benefit_pct",
        "socioeconomic",
        "Income_support",
        "People receiving an unemployment benefit (JobSeeker Payment or Youth Allowance (other) +++",  # noqa: E501
        "% people receiving an unemployment benefit",
        "June 2025",
    ),
    _p(
        "long_term_unemployment_benefit_pct",
        "socioeconomic",
        "Income_support",
        "People receiving an unemployment benefit long-term +++",
        "% people receiving a JobSeeker Payment or Youth Allowance (other) long-term",
        "June 2025",
    ),
    _p(
        "low_income_welfare_families_pct",
        "socioeconomic",
        "Income_support",
        "Low income, welfare-dependent families (with children) +++",
        "% low income, welfare-dependent families (with children)",
        "June 2025",
    ),
    _p(
        "health_care_card_pct",
        "socioeconomic",
        "Income_support",
        "Health Care Card holders +++",
        "% Health Care Card holders",
        "June 2025",
    ),
    _p(
        "pensioner_concession_card_pct",
        "socioeconomic",
        "Income_support",
        "Pensioner Concession Card holders +++",
        "% Pensioner Concession Card holders",
        "June 2025",
    ),
    _p(
        "single_parent_families_pct",
        "socioeconomic",
        "Families",
        "Single parent families with children aged less than 15 years",
        "% single parent families",
        "2021",
    ),
    _p(
        "jobless_families_pct",
        "socioeconomic",
        "Families",
        "Jobless families with children aged less than 15 years",
        "% jobless families",
        "2021",
    ),
    _p(
        "early_school_leavers_asr",
        "socioeconomic",
        "Education",
        "People who left school at Year 10 or below, or did not go to school",
        "ASR per 100",
        "2021",
    ),
    _p(
        "ft_secondary_at_16_pct",
        "socioeconomic",
        "Education",
        "Full-time participation in secondary school education at age 16",
        "% full-time participation at age 16",
        "2021",
    ),
    _p(
        "learning_or_earning_pct",
        "socioeconomic",
        "Learning_Earning",
        "Learning or Earning at ages 15 to 24",
        "% Learning or Earning at ages 15 to 24",
        "2021",
    ),
    _p(
        "crowded_dwellings_pct",
        "socioeconomic",
        "Housing_Transport",
        "Household crowding and suitability",
        "% people living in crowded dwellings",
        "2021",
    ),
    _p(
        "social_housing_pct",
        "socioeconomic",
        "Housing_Transport",
        "People living in rental housing",
        "% people living in social housing",
        "2021",
    ),
    _p(
        "rent_assistance_pct",
        "socioeconomic",
        "Housing_Transport",
        "Households receiving rent assistance from the Australian Government +++",
        "% households in dwellings receiving rent assistance",
        "June 2025",
    ),
    _p(
        "housing_stress_pct",
        "socioeconomic",
        "Housing_Transport",
        "Housing stress",
        "% Low income households under financial stress from mortgage or rent",
        "2021",
    ),
    _p(
        "homelessness_asr",
        "socioeconomic",
        "Homelessness",
        "Estimated number of people experiencing homelessness",
        "ASR per 10,000",
        "2021",
    ),
    # Demographic
    _p(
        "age_0_14_pct",
        "demographic",
        "Age_distribution_Persons_broad",
        "Persons, 0-14 years",
        "%",
        "2024",
    ),
    _p(
        "age_65_plus_pct",
        "demographic",
        "Age_distribution_Persons_broad",
        "Persons, 65 years and over",
        "%",
        "2024",
    ),
    _p(
        "age_85_plus_pct",
        "demographic",
        "Age_distribution_Persons_broad",
        "Persons, 85 years and over",
        "%",
        "2024",
    ),
    _p(
        "aboriginal_pct",
        "demographic",
        "Indigenous_proportion",
        "Aboriginal population as a proportion of total population",
        "Aboriginal population as proportion of total population (%)",
        "2021",
    ),
    _p(
        "born_nes_pct",
        "demographic",
        "Birthplace_NES_residents",
        "People born in predominantly non-English-speaking countries",
        "% born in non-English-speaking countries",
        "2021",
    ),
    _p(
        "poor_english_pct",
        "demographic",
        "Birthplace_NES_residents",
        "People born overseas reporting poor proficiency in English",
        "% born overseas who speak English not well or not at all",
        "2021",
    ),
    # Access
    _p(
        "no_motor_vehicle_pct",
        "access",
        "Housing_Transport",
        "Dwellings with no motor vehicle",
        "% dwellings with no motor vehicle",
        "2021",
    ),
    _p(
        "private_health_insurance_pct",
        "access",
        "Private_health_insurance",
        "Private health insurance",
        "% people with private health insurance",
        "2023-24",
        note="reported with and without, see docs/data.md",
    ),
    _p(
        "aged_care_places_per_1000",
        "access",
        "Aged_care_places",
        "Residential aged care places",
        "Residential care places per 1,000 population aged 70 years and over",
        "June 2025",
    ),
    Feature(
        "ra_inner_regional_share",
        "access",
        "A",
        "abs_ra_allocation",
        year="2021",
        note="population share, from SA1 RA allocation and SA1 usual resident population",
    ),
    Feature("ra_outer_regional_share", "access", "A", "abs_ra_allocation", year="2021"),
    Feature(
        "ra_remote_share",
        "access",
        "A",
        "abs_ra_allocation",
        year="2021",
        note="Remote and Very remote combined",
    ),
    Feature(
        "km_to_public_hospital",
        "access",
        "A",
        "myhospitals_reporting_units",
        year="2026",
        note="population-weighted over SA2 centroids, open public hospitals",
    ),
    Feature(
        "km_to_ed_reporting_hospital",
        "access",
        "A",
        "myhospitals_ed_presentations",
        year="2023-24",
        note="only hospitals in the national ED collection",
    ),
    # GP use
    Feature(
        "gp_any_claim_pct",
        "gp_use",
        "A",
        "aihw_mbs_sa3",
        year="2023-24",
        note="GP attendances (total): % of people who had the service",
    ),
    Feature(
        "gp_services_per_100",
        "gp_use",
        "A",
        "aihw_mbs_sa3",
        year="2023-24",
        note="GP attendances (total): services per 100 people, crude",
    ),
    Feature(
        "gp_after_hours_per_100",
        "gp_use",
        "A",
        "aihw_mbs_sa3",
        year="2023-24",
        note="GP subtotal - After-hours: services per 100 people, crude",
    ),
    # Prevention (primary care reach, not hospital use)
    _p(
        "immunised_1yr_pct",
        "prevention",
        "Child_youth_health",
        "Children fully immunised at 1 year of age",
        "% children fully immunised at 1 year of age",
        "2023",
    ),
    _p(
        "immunised_2yr_pct",
        "prevention",
        "Child_youth_health",
        "Children fully immunised at 2 years of age",
        "% children fully immunised at 2 years of age",
        "2023",
    ),
    _p(
        "immunised_5yr_pct",
        "prevention",
        "Child_youth_health",
        "Children fully immunised at 5 years of age",
        "% children fully immunised at 5 years of age",
        "2023",
    ),
    _p(
        "bowel_screening_pct",
        "prevention",
        "Screening",
        "Participation in the NBCSP, persons",
        "Per cent",
        "2022 and 2023",
    ),
    _p(
        "breast_screening_pct",
        "prevention",
        "Screening",
        "Breast screening participation, females aged 50 to 74 years",
        "Per cent",
        "2021 and 2022",
    ),
    _p(
        "no_early_antenatal_pct",
        "prevention",
        "Mothers_babies",
        "Antenatal visits",
        "% Women who did not attend antenatal care within the first 10 weeks",
        "2021 to 2023",
    ),
)

# Tier B is defined in Phase 2 once its column labels are resolved; it is a sensitivity check only.
TIER_B: tuple[Feature, ...] = ()

ALLOWLIST: tuple[Feature, ...] = TIER_A + TIER_B


class LeakageError(ValueError):
    """A feature definition breaks one of the leakage rules."""


def validate_allowlist(features: tuple[Feature, ...] = ALLOWLIST) -> None:
    names = [f.name for f in features]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise LeakageError(f"duplicate feature names: {dupes}")
    keys = [(f.sheet, f.block, f.measure) for f in features if f.source == PHIDU_SOURCE]
    if len(keys) != len(set(keys)):
        raise LeakageError("two features point at the same PHIDU column")
    for f in features:
        if f.source == TARGET_SOURCE:
            raise LeakageError(f"{f.name}: features may not come from the target source")
        if f.group not in GROUPS or f.tier not in TIERS:
            raise LeakageError(f"{f.name}: unknown group {f.group!r} or tier {f.tier!r}")
        if f.source == PHIDU_SOURCE:
            if not (f.sheet and f.block and f.measure):
                raise LeakageError(f"{f.name}: PHIDU features need sheet, block and measure")
            if DENY_SHEET_RE.match(f.sheet):
                raise LeakageError(f"{f.name}: sheet {f.sheet!r} holds hospital-use data")
            if DENY_MEASURE_RE.search(f.measure):
                raise LeakageError(f"{f.name}: measure {f.measure!r} is not a rate or percentage")
        if f.group == "health_status" and f.tier != "B":
            raise LeakageError(f"{f.name}: health-status features are Tier B (sensitivity) only")


def feature_names(tiers: tuple[str, ...] = ("A",), exclude: tuple[str, ...] = ()) -> list[str]:
    """Names of the allowlisted features in the given tiers, in a stable order."""
    validate_allowlist()
    return [f.name for f in ALLOWLIST if f.tier in tiers and f.name not in exclude]


def check_columns(columns: list[str], tiers: tuple[str, ...] = ("A", "B")) -> None:
    """Raise LeakageError if a model matrix has any column that is not on the allowlist."""
    allowed = set(feature_names(tiers))
    extra = sorted(set(columns) - allowed)
    if extra:
        raise LeakageError(f"columns not on the feature allowlist: {extra}")
