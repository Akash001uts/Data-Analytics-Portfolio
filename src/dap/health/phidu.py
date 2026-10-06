"""Read the layout of PHIDU Social Health Atlas sheets.

Each data sheet has block titles in row 1, time periods in row 4 and measure labels in row 5.
Area rows start at row 6: the PHAs first, then an "AUSTRALIA+" row, then state, SA4 and SA3 totals.
Thirty PHA codes are also SA3 codes (both have five digits), so rows must be split on the
AUSTRALIA+ marker, never on code length.
"""

from dataclasses import dataclass
from pathlib import Path

import openpyxl

from dap.health.features import PHIDU_SOURCE, Feature

TITLE_ROW, PERIOD_ROW, MEASURE_ROW = 1, 4, 5
TOTALS_MARKER = "AUSTRALIA+"


def norm(value: object) -> str:
    return " ".join(str(value or "").split())


@dataclass(frozen=True)
class Column:
    index: int  # zero-based
    block: str
    period: str
    measure: str


def read_header(ws) -> list[Column]:
    rows = list(ws.iter_rows(min_row=1, max_row=MEASURE_ROW, values_only=True))
    title, period, measure = rows[TITLE_ROW - 1], rows[PERIOD_ROW - 1], rows[MEASURE_ROW - 1]
    cols, block, block_period = [], "", ""
    for j in range(2, len(measure)):
        if j < len(title) and title[j]:
            block, block_period = norm(title[j]), ""
        if j < len(period) and period[j]:
            block_period = norm(period[j])
        if measure[j]:
            cols.append(Column(j, block, block_period, norm(measure[j])))
    return cols


def locate(feature: Feature, header: list[Column]) -> Column:
    """Find the single column for a PHIDU feature, or raise KeyError."""
    if feature.source != PHIDU_SOURCE:
        raise ValueError(f"{feature.name} is not a PHIDU feature")
    hits = [c for c in header if c.block == feature.block and c.measure == feature.measure]
    if len(hits) != 1:
        raise KeyError(
            f"{feature.name}: {len(hits)} columns match block {feature.block!r} and measure "
            f"{feature.measure!r} in sheet {feature.sheet!r}"
        )
    return hits[0]


def resolve_features(workbook: Path, features: tuple[Feature, ...]) -> dict[str, Column]:
    """Map each PHIDU feature to its column. Raises KeyError naming every missing feature."""
    wb = openpyxl.load_workbook(workbook, read_only=True)
    try:
        headers: dict[str, list[Column]] = {}
        found, problems = {}, []
        for f in features:
            if f.source != PHIDU_SOURCE:
                continue
            if f.sheet not in wb.sheetnames:
                problems.append(f"{f.name}: no sheet {f.sheet!r}")
                continue
            if f.sheet not in headers:
                headers[f.sheet] = read_header(wb[f.sheet])
            try:
                found[f.name] = locate(f, headers[f.sheet])
            except KeyError as e:
                problems.append(str(e).strip("'\""))
        if problems:
            raise KeyError("; ".join(problems))
        return found
    finally:
        wb.close()
