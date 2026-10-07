# The data behind this project

This is the detailed version of the data side of the project: every source I use, why I picked it, and every check I
ran before building anything. If you just want the overview, start with the [project README](README.md).
Every file is listed with its URL, size and SHA256 in [`data/manifest.yaml`](../../data/manifest.yaml).
None of the raw data is in git. `dap health fetch` downloads it and checks it.

The checks below were first run on 6 October 2026. PHIDU published a September 2026 release the same
week, so I re-pinned it on 7 October and re-ran every PHIDU-based number: none of them changed.

## Question

Which areas of Australia have more potentially preventable hospitalisations (PPH) than their social, demographic
and access profile predicts?

## Why SA3 and not PHA

I model at **Statistical Area Level 3 (SA3, ASGS 2021)**. I only use the finer Population Health Areas (PHAs) for
one descriptive map.

| | PHA (PHIDU Social Health Atlas) | SA3 (AIHW HPF 76) |
|---|---|---|
| Areas | 1,165 | 340 (336 with PHIDU features) |
| PPH years | 2020/21 only | 2017-18 to 2023-24 |
| Hospitals covered | **Public only** | Public and private |
| Suppressed (all PPH) | 7 | 9 of 336 in 2023-24 |
| Licence | CC BY-NC-SA 3.0 AU | CC BY 4.0 |

Three things decided it for me:

1. **The PHA target is public hospitals only, and the gap tracks private health insurance.** PHIDU also publishes SA3
   totals, so I compared its public-only 2020/21 rate with AIHW's all-hospital 2020-21 rate for the 319 SA3s where both
   are published. They agree closely (Pearson r = 0.96, Spearman 0.90), but the public share of PPH falls from a median of
   0.94 in the least-insured fifth of SA3s to 0.67 in the most-insured fifth (r = -0.68 with the percentage of adults
   holding private cover). A residual map built on the public-only rate would partly be a map of private insurance,
   which would undermine the main result. The ratio isn't a pure public share, though: it exceeds 1 in 22 SA3s, so the two
   extracts also differ in other ways (PHIDU's errata re-release, AIHW's SA2 2016 to 2021 mapping), but the gradient
   with insurance is much bigger than that noise.
2. **2020/21 is a COVID year.** The national age-standardised PPH rate was 2,735 per 100,000 in 2018-19, 2,371 in
   2020-21 and 2,617 in 2023-24. Vaccine-preventable admissions fell from 245 to 101 and had recovered to 252 by
   2023-24. The AIHW data lets me use 2023-24 and check it against pre-COVID 2018-19.
3. **GP use only exists at SA3.** The current PHIDU release has no GP or Medicare indicator at any level, so the only
   GP-use data is AIHW PHC 19, which is published by SA3.

The cost is sample size: 327 SA3s with a published 2023-24 rate, against about 1,158 PHAs. That's enough for the
linear and spatial lag models. For LightGBM I kept the trees shallow and regularised, and judged it on its
spatial CV score instead of assuming it would win.

### Nesting and CV groups

Spatial cross-validation groups SA3s by SA4.

- SA2 codes embed their SA3 (first 5 digits) and SA4 (first 3). I checked the PHIDU concordance against the ABS SA2
  allocation file rather than relying on the code prefix.
- All 2,450 concordance SA2s are in the ABS file. **No PHA spans more than one SA3, SA4 or state**, so every PHA maps to
  exactly one SA4 and PHA values can be aggregated to SA3 without splitting.
- In the ABS allocation file, every one of the 340 spatial SA3s sits in exactly one SA4, and the SA4 code always equals
  the first three digits of the SA3 code.
- The 23 ABS SA2s missing from the concordance are 18 non-spatial codes (migratory and no usual address, two for each
  state and territory), "Outside Australia", and the 4 Other Territories SA2s (Christmas Island, Cocos (Keeling) Islands,
  Jervis Bay, Norfolk Island). AIHW publishes those 4 as SA3s 90101 to 90104, but PHIDU has no features for them, so
  I dropped them.
- That leaves 336 SA3s in 88 SA4s, and 327 SA3s with a target, which gives a median of 3 SA3s per SA4 (range 1 to 10).
  Holding out whole states (8 groups) would be an even harsher check, which I haven't run.

### The missing areas aren't random

The 9 SA3s without a 2023-24 rate are Illawarra Catchment Reserve, Lord Howe Island, Blue Mountains - South,
West Pilbara, Barkly, East Arnhem, Canberra East, Molonglo and Uriarra - Namadgi. Some are near-empty areas, but
Barkly, East Arnhem and West Pilbara are remote areas with very high rates: PHIDU's public-only 2020/21 SA3 figures
are about 18,600 (Barkly), 11,300 (East Arnhem) and 3,100 (West Pilbara) per 100,000, against an SA3 median of about
1,900. I had to drop them from
training and testing, and I list it as a limitation, because the model never sees the most remote part of the
distribution.

## Sources

| ID | What | Edition and years | Use | Licence |
|---|---|---|---|---|
| `aihw_pph_sa3` | AIHW HPF 76, Table 4: PPH by SA3 | SA3 2021, 2017-18 to 2023-24, public and private | Target | CC BY 4.0 |
| `phidu_sha_pha` | PHIDU Social Health Atlas, PHA workbook (September 2026 release), SA3 total rows | PHA and SA3 2021; indicator years vary (mostly 2021 Census) | Features; PHA map | CC BY-NC-SA 3.0 AU |
| `phidu_sa2_pha_concordance` | SA2 to PHA concordance | 2021 | Nesting checks | CC BY-NC-SA 3.0 AU |
| `aihw_mbs_sa3` | AIHW PHC 19, Table 3: Medicare-subsidised services by SA3 | SA3 2021, 2017-18 to 2024-25 | GP-use features | CC BY 4.0 |
| `abs_sa2_boundaries` | ABS SA2 boundaries, GDA2020 | ASGS Edition 3 (2021) | Geometry, SA3 dissolve, centroids | CC BY 4.0 |
| `abs_sa2_allocation` | ABS SA2 allocation file | ASGS Edition 3 | Nesting, CV groups | CC BY 4.0 |
| `abs_ra_allocation` | ABS Remoteness Areas allocation (SA1 to RA) | RA 2021 | Remoteness features | CC BY 4.0 |
| `abs_seifa_sa1` | SEIFA 2021 SA1 indexes, with usual resident population | 2021 | Population weights for remoteness | CC BY 4.0 |
| `abs_seifa_sa2` | SEIFA 2021 SA2 indexes | 2021 | Cross-check of PHIDU IRSD, SA2 weights | CC BY 4.0 |
| `myhospitals_reporting_units` | MyHospitals API reporting units | Live, pinned by counts | Distance to nearest public hospital | CC BY 3.0 |
| `myhospitals_ed_presentations` | MyHospitals API, measure MYH0035 | 2023-24 data sets 10511 to 10515 | Distance to nearest ED-reporting hospital | CC BY 3.0 |

### Target

The AIHW **Total PPH, all persons, age-standardised rate per 100,000, 2023-24**, weighted by the published estimated
resident population. Sensitivity checks use 2018-19 (pre-COVID) and the chronic and acute subtotals.

AIHW suppresses a rate when the count is 1 to 4 or the population is under 300 (confidentiality), and when the count is
under 20 or the population under 2,500 (volatility). The smallest published 2023-24 SA3 count is 204, so every SA3 I
keep has a stable rate. AIHW recorded residence on SA2 2016 up to 2021-22 and mapped it to SA2 2021, so the 2018-19
check carries a small boundary-mapping error. AIHW also notes that some ACT private hospitals are missing before
2019-20, so I left the ACT SA3s out of the 2018-19 check.

### Things to know about each source

- **PHIDU workbook.** Each sheet lists the 1,165 PHAs first, then an `AUSTRALIA+` row, then state, SA4 and SA3 totals.
  Thirty PHA codes are also SA3 codes (both are five digits), so my parser splits on the `AUSTRALIA+` row, not
  on code length. Each state also has a pseudo-area coded `<state>9999` ("ABS cell adjustment" in
  Census sheets, "Unknown <state>" in admissions sheets), which my parser sets aside. Values use `#` (population under 100), `..` (not applicable), `n.p.` and `n.a.`, and all of these
  become missing. The `/current/` URL is overwritten at each release, so the fetch fails loudly on a checksum mismatch;
  archived releases are kept as yearly zips.
- **PHIDU PPH (PHA map only).** It covers public hospitals only, for 2020/21, re-released in December 2025 after an
  error in the December 2023 data. Seven PHAs are unpublished. Vaccine-preventable PPH is too sparse to use at PHA level
  (237 unpublished, 552 PHAs with counts under 20).
- **GP use.** Allocated by Medicare enrolment postcode rather than SA2, with 10 SA3s unpublished. "Services per 100
  people" is crude, not age-standardised, at SA3, so I checked age structure separately.
- **Remoteness.** For each SA3: the population share in each Remoteness Area, built from the SA1 to RA allocation and
  SA1 usual resident population. SA1s that SEIFA doesn't score (usually because very few people live there) carry no
  weight. By count, a median 97% of each SA3's SA1s are matched, and the lowest is East Pilbara at 75%, so the shares
  there are a little less certain.
- **Hospital distance.** MyHospitals lists 1,166 hospitals: 692 are open public hospitals, and one of those has no
  coordinates. Only 293 hospitals report ED presentations in 2023-24, because the national ED collection mainly covers
  larger EDs. Small rural hospitals that do treat emergencies are missing, so I call that feature "distance to nearest
  ED-reporting hospital" and add "distance to nearest open public hospital" next to it. Both are population-weighted
  over SA2 centroids. The API is live, so I pin data set IDs and record counts instead of hashing the response.

## Feature allowlist

The model inputs are an explicit allowlist of (sheet, indicator) pairs from the PHIDU SA3 rows, plus the access
features I worked out myself. Anything that isn't on the list can't reach a model. `src/dap/health/features.py` holds the list, and
`tests/test_features.py` and `tests/test_allowlist_workbook.py` enforce the rules below. The second test
also checks that every PHIDU feature resolves to exactly one column of the real workbook, so if a future
release renames a sheet or relabels a column, the test fails instead of the model quietly losing an input.

### Rules the tests enforce

1. **Hard denylist by sheet.** No feature may come from a sheet matching `^(Hosp_|Admiss|ED_)`. That covers admissions by
   hospital type and by diagnosis (which include PPH directly), injury admissions, procedures, same-day renal dialysis,
   all four `Admissions_prevent_diag_*` sheets and every ED presentation sheet.
2. **Nothing from the target source.** No column from `aihw_pph_sa3` may be a feature.
3. **Rates and percentages only.** Never `Number`, `SR`, `Sig.`, confidence limits, RRMSE, index minimum or maximum, or
   rank columns. Population (ERP) is a weight, not a feature.
4. **Allowlist at indicator level.** Some sheets mix drivers and outcomes. For example, `Child_youth_health` has
   immunisation next to infant and youth mortality, so only the listed indicators are allowed.

### Tier A: core features (used by default)

| Group | Sheet | Indicator (measure) | Year |
|---|---|---|---|
| Socioeconomic | `IRSD` | IRSD index score | 2021 |
| | `Labour_force` | % unemployed; % labour force participation | June 2025 |
| | `Income_support` | % disability support pension; % receiving an unemployment benefit; % long-term unemployment benefit; % low-income welfare-dependent families; % Health Care Card holders; % Pensioner Concession Card holders | June 2025 |
| | `Families` | % single-parent families; % jobless families | 2021 |
| | `Education` | Early school leavers (ASR per 100); % full-time secondary at 16 | 2021 |
| | `Learning_Earning` | % learning or earning at 15 to 24 | 2021 |
| | `Housing_Transport` | % people in crowded dwellings; % people in social housing; % households receiving rent assistance; % low-income households under financial stress | 2021, June 2025 |
| | `Homelessness` | Homelessness (ASR per 10,000) | 2021 |
| Demographic | `Age_distribution_Persons_broad` | % aged 0 to 14; % 65 and over; % 85 and over | 2024 |
| | `Indigenous_proportion` | % Aboriginal population | 2021 |
| | `Birthplace_NES_residents` | % born in non-English-speaking countries; % poor English proficiency | 2021 |
| Access | `Housing_Transport` | % dwellings with no motor vehicle | 2021 |
| | `Private_health_insurance` | % adults with private health insurance (flagged, see below) | 2023-24 |
| | `Aged_care_places` | Residential aged care places per 1,000 aged 70 and over | June 2025 |
| | derived (ABS RA) | Population share in Inner regional, Outer regional, and Remote or Very remote areas | 2021 |
| | derived (MyHospitals) | Population-weighted km to nearest open public hospital; to nearest ED-reporting hospital | 2023-24 |
| GP use | derived (AIHW PHC 19) | GP attendances: % of people with a claim; services per 100 people; after-hours GP services per 100 people | 2023-24 |
| Prevention | `Child_youth_health` | % children fully immunised at 1, 2 and 5 years | 2023 |
| | `Screening` | Bowel screening participation (persons); breast screening participation | 2022 and 2023 |
| | `Mothers_babies` | % women with no antenatal visit in the first 10 weeks | 2021 to 2023 |

I kept private health insurance in because it plausibly changes where and whether people are admitted, and with an
all-hospitals target that's a real effect, not a reporting artefact. I also ran the models without it, and the
results are in the sensitivity checks in [notebook 02](notebooks/02_spatial_models.ipynb).

The prevention group goes a bit beyond what I first planned to use (socioeconomic, demographic, access, GP use and
distance). I added it because it measures how far primary care reaches, not hospital use.

### Tier B: health status (sensitivity only, off by default)

17 features, all age-standardised rates or percentages: Census long-term conditions (arthritis, asthma, diabetes,
heart disease, kidney disease, lung conditions, mental health conditions, stroke, three or more conditions); modelled
fair or poor self-assessed health; modelled adult risk factors (psychological distress, high blood pressure, obesity,
smoking, risky drinking, physical inactivity); and profound or severe disability.

In the Census condition sheets the condition name sits in the second header row, under a sheet-wide caveat, so the
reader matches a feature's title against either header row and insists on exactly one column. PHIDU's modelled
estimates aren't published for 20 SA3s (13 of them have a target, mostly remote), so the Tier B check runs on fewer
areas, and leaves out some of the highest-rate ones.

I left these out of the main model because they sit on the causal path. Lots of chronic disease explains chronic
PPH without saying anything about primary care, and PHIDU's modelled estimates are themselves predicted from
socio-demographic data, so they'd double-count Tier A. I only use a Tier A + B model as a sensitivity check.

### Excluded

| Reason | Sheets or columns |
|---|---|
| Hospital use (denylist) | `Hosp_type_sex`, all `Admiss*` sheets, all `ED_*` sheets |
| Outcomes, not drivers | `Median_age_death`, `Premature_mortality_*`, `Avoidable_mortality_*`, `Years_life_lost_*`, `Cancer_incidence_*`, infant and youth mortality in `Child_youth_health`, low birthweight in `Mothers_babies` |
| Other service use, out of scope | `CMHCS_*` (community mental health), `CHSP` (home support), `NDIS_*` |
| Redundant or weak | 5-year age groups, Aboriginal age distributions, population projections (2025, 2030), migrant streams, preschool, VET and university measures, unpaid child care, volunteering, Aboriginal-only columns |
| Not in this release | DVA age pensioners, internet access, breastfeeding, modelled diabetes and other chronic disease estimates (PHIDU marks these "not included in this release") |
| Wrong value type | `Number`, `SR`, `Sig.`, confidence limits, RRMSE, ERP and population columns |

### Timing

Most features describe 2021 (Census). Income support and labour force are June 2025, slightly after the 2023-24
target year. I treat all of them as things about an area that change slowly, and list the mismatch as a limitation. GP use
comes from the same year as the target.

## Licences and attribution

- Code in this repository is MIT licensed.
- **Outputs derived from PHIDU data** (cleaned tables, figures, the map and `reports/results.json`) are released under
  **CC BY-NC-SA 3.0 AU**, as PHIDU's ShareAlike terms require, and are for non-commercial use.
- PHIDU attribution for modified material: "Based on Public Health Information Development Unit (PHIDU), Torrens
  University Australia material from: Social Health Atlas of Australia: Population Health Areas (online) 2026. Accessed
  7 October 2026, https://phidu.torrens.edu.au/social-health-atlases/data".
- AIHW data tables are CC BY 4.0 (aihw.gov.au/copyright). The MyHospitals API states CC BY 3.0 in its own metadata.
- ABS boundaries, allocation files and SEIFA are CC BY 4.0.

## Limitations

- It's area-level data, so the results describe areas, not patients.
- The suppressed SA3s aren't missing at random, and they include remote areas with high need.
- ED distance only covers hospitals in the national ED collection.
- Feature years range from 2021 to 2025 against a 2023-24 target.
- GP use is allocated by postcode, which only approximately matches SA3 boundaries.
- PPH counts episodes, not people, so one person can be admitted several times.
