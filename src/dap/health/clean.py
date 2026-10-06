"""Clean each source and join them into one table with a row per SA3.

Each source has a `read_*` function (file in, raw-ish DataFrame out) and a pure transform that the
tests can run on small in-memory tables. `build_sa3_table` puts them together.
"""

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import openpyxl
import pandas as pd

from dap.health.features import ALLOWLIST, PHIDU_SOURCE, Feature
from dap.health.phidu import locate, read_header, read_rows, to_number

log = logging.getLogger(__name__)

ALBERS = "EPSG:3577"  # GDA94 Australian Albers, metres; good enough for distances at this scale
TARGET_YEAR = "2023-24"
PRE_COVID_YEAR = "2018-19"
TARGET_CATEGORIES = {
    "Total PPH": "pph",
    "Total chronic": "pph_chronic",
    "Total acute": "pph_acute",
    "Total vaccine-preventable": "pph_vaccine",
}
SUPPRESSED = {"n.p.", "..", "#", "n.a.", "np", ""}


def _dash(s: pd.Series) -> pd.Series:
    """AIHW writes years with an en dash (2023–24); normalise to a hyphen."""
    return s.astype(str).str.replace("–", "-", regex=False).str.strip()


# ---------------------------------------------------------------------------------------------
# PHIDU features (SA3 total rows)


def phidu_features(
    workbook: Path, features: tuple[Feature, ...], sa3_codes: set[str]
) -> pd.DataFrame:
    """One column per PHIDU feature, one row per SA3, read from the totals below AUSTRALIA+."""
    wb = openpyxl.load_workbook(workbook, read_only=True)
    try:
        cols: dict[str, pd.Series] = {}
        by_sheet: dict[str, list[Feature]] = {}
        for f in features:
            if f.source == PHIDU_SOURCE:
                by_sheet.setdefault(f.sheet, []).append(f)
        for sheet, fs in by_sheet.items():
            ws = wb[sheet]
            header = read_header(ws)
            rows = read_rows(ws)
            missing = sorted(sa3_codes - set(rows.totals))
            if missing:
                raise KeyError(f"{sheet}: no SA3 total rows for {missing[:5]}")
            for f in fs:
                col = locate(f, header)
                cols[f.name] = pd.Series(
                    {code: to_number(rows.totals[code][col.index]) for code in sorted(sa3_codes)},
                    dtype="float64",
                )
    finally:
        wb.close()
    out = pd.DataFrame(cols)
    out.index.name = "sa3_code"
    return out[[f.name for f in features if f.name in cols]]


# ---------------------------------------------------------------------------------------------
# Target: AIHW PPH by SA3


def read_aihw_pph(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="Table 4", header=1, dtype=str)


def tidy_pph(raw: pd.DataFrame) -> pd.DataFrame:
    """Long table of SA3 x year x category with numeric rates and an explicit suppressed flag."""
    df = raw.rename(columns=lambda c: str(c).strip())
    df = df[df["SA3 code"].astype(str).str.fullmatch(r"\d{5}", na=False)]
    df = df[df["Demographic group"].astype(str).str.strip() == "All persons"]
    cond = "Potentially preventable hospitalisation (PPH) condition"
    df = df[df[cond].isin(TARGET_CATEGORIES)]
    asr_raw = df["Hospitalisations per 100,000 people (age-standardised)"].astype(str).str.strip()
    out = pd.DataFrame(
        {
            "sa3_code": df["SA3 code"].str.strip(),
            "year": _dash(df["Year"]),
            "category": df[cond].map(TARGET_CATEGORIES),
            "aihw_sa3_group": df["SA3 group"].str.strip(),
            "asr": pd.to_numeric(asr_raw, errors="coerce"),
            "count": pd.to_numeric(df["Number of hospitalisations"], errors="coerce"),
            "erp": pd.to_numeric(df["Estimated resident population"], errors="coerce"),
            "suppressed": asr_raw.isin(SUPPRESSED) | asr_raw.eq("nan"),
        }
    )
    bad = out[~out.suppressed & out.asr.isna()]
    if len(bad):
        raise ValueError(f"unparsed AIHW rates: {bad.head().to_dict('records')}")
    return out.reset_index(drop=True)


def target_columns(tidy: pd.DataFrame, year: str = TARGET_YEAR) -> pd.DataFrame:
    """Wide target table: the main rate and count, subtotals, ERP, and the pre-COVID rate."""
    cur = tidy[tidy.year == year]
    wide = cur.pivot(index="sa3_code", columns="category", values="asr")
    wide.columns = [f"{c}_asr" for c in wide.columns]
    total = cur[cur.category == "pph"].set_index("sa3_code")
    wide["pph_count"] = total["count"]
    wide["erp"] = total["erp"]
    wide["aihw_sa3_group"] = total["aihw_sa3_group"]
    pre = tidy[(tidy.year == PRE_COVID_YEAR) & (tidy.category == "pph")].set_index("sa3_code")
    pre_asr = pre["asr"].copy()
    # AIHW: some ACT private hospitals are missing before 2019-20, so the ACT (state 8) is left out.
    pre_asr[pre_asr.index.str.startswith("8")] = np.nan
    wide[f"pph_asr_{PRE_COVID_YEAR.replace('-', '_')}"] = pre_asr
    wide["has_target"] = wide["pph_asr"].notna()
    return wide


# ---------------------------------------------------------------------------------------------
# GP use: AIHW Medicare-subsidised services by SA3

GP_SERVICES = {
    "GP attendances (total)": {
        "Percentage of people who had the service (%)": "gp_any_claim_pct",
        "Services per 100 people": "gp_services_per_100",
    },
    "GP subtotal - After-hours": {"Services per 100 people": "gp_after_hours_per_100"},
}


def read_mbs(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="Table 3", header=1, dtype=str)


def gp_use(raw: pd.DataFrame, year: str = TARGET_YEAR) -> pd.DataFrame:
    df = raw.rename(columns=lambda c: str(c).strip())
    df = df[
        df["SA3 code"].astype(str).str.fullmatch(r"\d{5}", na=False)
        & (_dash(df["Year"]) == year)
        & (df["Demographic group"].astype(str).str.strip() == "All persons")
    ]
    df = df.assign(Service=df["Service"].astype(str).str.strip())
    out = {}
    for service, measures in GP_SERVICES.items():
        rows = df[df.Service == service]
        rows = rows.set_index(rows["SA3 code"].str.strip())
        if rows.index.duplicated().any():
            raise ValueError(f"duplicate SA3 rows for {service!r} in {year}")
        if rows.empty:
            raise KeyError(f"no MBS rows for {service!r} in {year}")
        for measure, name in measures.items():
            out[name] = pd.to_numeric(rows[measure], errors="coerce")
    res = pd.DataFrame(out)
    res.index.name = "sa3_code"
    return res


# ---------------------------------------------------------------------------------------------
# Remoteness: population share of each SA3 in each Remoteness Area

RA_CLASS = {
    "0": "major_city",
    "1": "inner_regional",
    "2": "outer_regional",
    "3": "remote",
    "4": "remote",
}


def read_sa1_population(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Table 1", header=None, dtype=str, skiprows=6)
    out = pd.DataFrame(
        {"sa1_code": raw[0].str.strip(), "pop": pd.to_numeric(raw[9], errors="coerce")}
    )
    return out[out.sa1_code.str.fullmatch(r"\d{11}", na=False)]


def read_ra(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, dtype=str)[["SA1_CODE_2021", "RA_CODE_2021"]]


def remoteness_shares(ra: pd.DataFrame, sa1_pop: pd.DataFrame) -> pd.DataFrame:
    """Share of each SA3's population in Inner regional, Outer regional and Remote/Very remote.

    The SA3 is the first five digits of the SA1 code. SA1s with no SEIFA population (very small
    or unpublished) carry no weight. Major cities is the left-out category.
    """
    df = ra.rename(columns={"SA1_CODE_2021": "sa1_code", "RA_CODE_2021": "ra"}).merge(
        sa1_pop, on="sa1_code", how="inner"
    )
    df["cls"] = df.ra.str[1].map(RA_CLASS)
    df = df.dropna(subset=["cls"])
    df["sa3_code"] = df.sa1_code.str[:5]
    pop = df.pivot_table(index="sa3_code", columns="cls", values="pop", aggfunc="sum", fill_value=0)
    share = pop.div(pop.sum(axis=1), axis=0)
    out = pd.DataFrame(
        {
            "ra_inner_regional_share": share.get("inner_regional", 0.0),
            "ra_outer_regional_share": share.get("outer_regional", 0.0),
            "ra_remote_share": share.get("remote", 0.0),
        }
    )
    out.index.name = "sa3_code"
    return out


# ---------------------------------------------------------------------------------------------
# Geometry and hospital distances


def read_sa2(boundaries_zip: Path, seifa_sa2: Path) -> gpd.GeoDataFrame:
    sa2 = gpd.read_file(f"zip://{Path(boundaries_zip).as_posix()}")
    sa2 = sa2[sa2.geometry.notna()].rename(
        columns={
            "SA2_CODE21": "sa2_code",
            "SA3_CODE21": "sa3_code",
            "SA3_NAME21": "sa3_name",
            "SA4_CODE21": "sa4_code",
            "SA4_NAME21": "sa4_name",
            "GCC_CODE21": "gcc_code",
            "GCC_NAME21": "gcc_name",
            "STE_CODE21": "state_code",
            "STE_NAME21": "state_name",
        }
    )
    raw = pd.read_excel(seifa_sa2, sheet_name="Table 1", header=None, dtype=str, skiprows=6)
    pop = pd.DataFrame(
        {"sa2_code": raw[0].str.strip(), "pop": pd.to_numeric(raw[10], errors="coerce")}
    )
    sa2 = sa2.merge(pop[pop.sa2_code.str.fullmatch(r"\d{9}", na=False)], on="sa2_code", how="left")
    sa2["pop"] = sa2["pop"].fillna(0)
    keep = ["sa2_code", "sa3_code", "sa3_name", "sa4_code", "sa4_name", "gcc_code", "gcc_name"]
    keep += ["state_code", "state_name"]
    return sa2[[*keep, "pop", "geometry"]]


def hospital_points(
    reporting_units: Path, ed_items: Path, ed_data_set_ids: list[int]
) -> gpd.GeoDataFrame:
    """Open public hospitals with coordinates, flagged if they reported ED presentations."""
    with Path(reporting_units).open(encoding="utf-8-sig") as f:
        units = json.load(f)["result"]
    with Path(ed_items).open(encoding="utf-8-sig") as f:
        items = json.load(f)["result"]
    ids = set(ed_data_set_ids)
    ed = {
        i["reporting_unit_summary"]["reporting_unit_code"]
        for i in items
        if i["data_set_id"] in ids and (i.get("value") or 0) > 0
    }
    rows = [
        {
            "code": u["reporting_unit_code"],
            "name": u["reporting_unit_name"],
            "reports_ed": u["reporting_unit_code"] in ed,
            "lon": u["longitude"],
            "lat": u["latitude"],
        }
        for u in units
        if u["reporting_unit_type"]["reporting_unit_type_code"] == "H"
        and not u["closed"]
        and not u["private"]
        and u.get("latitude") is not None
    ]
    df = pd.DataFrame(rows)
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")


def nearest_km(points: gpd.GeoDataFrame, targets: gpd.GeoDataFrame) -> pd.Series:
    """Straight-line km from each point to the nearest target, in Australian Albers."""
    p = points.to_crs(ALBERS)
    t = targets.to_crs(ALBERS)[["geometry"]]
    joined = gpd.sjoin_nearest(p[["geometry"]], t, distance_col="m", how="left")
    joined = joined[~joined.index.duplicated()]  # ties
    return joined["m"].reindex(points.index) / 1000


def population_weighted(sa2: pd.DataFrame, value: pd.Series) -> pd.Series:
    """SA3 mean of an SA2 value, weighted by SA2 population (plain mean if every weight is 0)."""
    df = pd.DataFrame({"sa3_code": sa2.sa3_code, "w": sa2["pop"], "v": value})

    def agg(g: pd.DataFrame) -> float:
        return float(np.average(g.v, weights=g.w)) if g.w.sum() > 0 else float(g.v.mean())

    return df.groupby("sa3_code")[["w", "v"]].apply(agg)


def hospital_distances(sa2: gpd.GeoDataFrame, hospitals: gpd.GeoDataFrame) -> pd.DataFrame:
    pts = sa2.copy()
    pts["geometry"] = sa2.to_crs(ALBERS).representative_point()
    pts = pts.set_crs(ALBERS, allow_override=True)
    out = pd.DataFrame(
        {
            "km_to_public_hospital": population_weighted(sa2, nearest_km(pts, hospitals)),
            "km_to_ed_reporting_hospital": population_weighted(
                sa2, nearest_km(pts, hospitals[hospitals.reports_ed])
            ),
        }
    )
    out.index.name = "sa3_code"
    return out


def sa3_geometry(sa2: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    cols = ["sa3_code", "sa3_name", "sa4_code", "sa4_name", "gcc_code", "gcc_name"]
    cols += ["state_code", "state_name"]
    geo = sa2[[*cols, "geometry"]].dissolve(by="sa3_code", as_index=False, aggfunc="first")
    return geo.set_index("sa3_code")


# ---------------------------------------------------------------------------------------------
# Putting it together


def build_sa3_table(sources: dict[str, Path], ed_data_set_ids: list[int]) -> gpd.GeoDataFrame:
    """Join every source into one GeoDataFrame indexed by SA3 code.

    The SA3s kept are those PHIDU publishes totals for (336: every spatial SA3 except the four
    Other Territories ones). SA3s without a published target stay in, flagged by `has_target`.
    """
    log.info("reading SA2 boundaries")
    sa2 = read_sa2(sources["abs_sa2_boundaries"], sources["abs_seifa_sa2"])
    conc = pd.read_excel(
        sources["phidu_sa2_pha_concordance"], sheet_name="2021 SA2 to PHA concordance", dtype=str
    )
    covered = set(conc.iloc[:, 0].dropna().str.strip())
    sa2 = sa2[sa2.sa2_code.isin(covered)]
    sa3_codes = set(sa2.sa3_code)

    log.info("dissolving %d SA2s into %d SA3s", len(sa2), len(sa3_codes))
    geo = sa3_geometry(sa2)
    log.info("reading PHIDU features")
    phidu = phidu_features(sources["phidu_sha_pha"], ALLOWLIST, sa3_codes)
    log.info("reading AIHW PPH")
    target = target_columns(tidy_pph(read_aihw_pph(sources["aihw_pph_sa3"])))
    log.info("reading MBS GP use")
    gp = gp_use(read_mbs(sources["aihw_mbs_sa3"]))
    log.info("computing remoteness shares")
    ra = remoteness_shares(
        read_ra(sources["abs_ra_allocation"]), read_sa1_population(sources["abs_seifa_sa1"])
    )
    log.info("computing hospital distances")
    hosp = hospital_points(
        sources["myhospitals_reporting_units"],
        sources["myhospitals_ed_presentations"],
        ed_data_set_ids,
    )
    dist = hospital_distances(sa2, hosp)

    table = geo.join([target, phidu, gp, ra, dist], how="left")
    table["has_target"] = table["has_target"].fillna(False).astype(bool)
    feature_order = [f.name for f in ALLOWLIST]
    meta = ["sa3_name", "sa4_code", "sa4_name", "gcc_code", "gcc_name", "state_code", "state_name"]
    meta += ["aihw_sa3_group", "erp"]
    targets = [c for c in table.columns if c.startswith("pph_")] + ["has_target"]
    table = table[[*meta, *targets, *feature_order, "geometry"]]
    return gpd.GeoDataFrame(table, geometry="geometry", crs=geo.crs).sort_index()


def summarise(table: pd.DataFrame) -> dict:
    """Counts for the data summary JSON; README numbers about the data come from here."""
    feats = [f.name for f in ALLOWLIST]
    missing = table[feats].isna().sum()
    return {
        "sa3_count": int(len(table)),
        "sa3_with_target": int(table.has_target.sum()),
        "sa4_count": int(table.sa4_code.nunique()),
        "sa4_count_with_target": int(table.loc[table.has_target, "sa4_code"].nunique()),
        "suppressed_sa3": sorted(table.loc[~table.has_target, "sa3_name"].tolist()),
        "target_year": TARGET_YEAR,
        "features_tier_a": len([f for f in ALLOWLIST if f.tier == "A"]),
        "features_tier_b": len([f for f in ALLOWLIST if f.tier == "B"]),
        "feature_missing_sa3": {k: int(v) for k, v in missing.items() if v},
    }


def national_pph(raw: pd.DataFrame) -> pd.DataFrame:
    """National age-standardised PPH rate by year and category, for the COVID trend chart."""
    df = raw.rename(columns=lambda c: str(c).strip())
    cond = "Potentially preventable hospitalisation (PPH) condition"
    df = df[
        (df["SA3 code"] == "001NAT")
        & (df["Demographic group"].astype(str).str.strip() == "All persons")
        & df[cond].isin(TARGET_CATEGORIES)
    ]
    return pd.DataFrame(
        {
            "year": _dash(df["Year"]).to_numpy(),
            "category": df[cond].map(TARGET_CATEGORIES).to_numpy(),
            "asr": pd.to_numeric(
                df["Hospitalisations per 100,000 people (age-standardised)"]
            ).to_numpy(),
        }
    )
