"""`dap health fetch`: download every source in the manifest and check it.

Pinned files must match their size and SHA256 exactly. Live MyHospitals API responses change, so
they are saved as retrieved and checked against the record counts from my first download, within a
tolerance.
"""

import json
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from dap.common.manifest import (
    Manifest,
    ManifestMismatch,
    Source,
    download,
    resolve_url,
    verify,
)

log = logging.getLogger(__name__)

Counter = Callable[[Any], dict[str, int]]


def _result(doc: Any) -> list[dict[str, Any]]:
    return doc["result"] if isinstance(doc, dict) else doc


def _is_hospital(unit: dict[str, Any]) -> bool:
    return unit["reporting_unit_type"]["reporting_unit_type_code"] == "H"


def count_reporting_units(doc: Any) -> dict[str, int]:
    units = _result(doc)
    hospitals = [u for u in units if _is_hospital(u)]
    return {
        "reporting_units": len(units),
        "hospitals": len(hospitals),
        "hospitals_open_public": sum(not h["closed"] and not h["private"] for h in hospitals),
        "hospitals_missing_coordinates": sum(h.get("latitude") is None for h in hospitals),
    }


def count_ed_presentations(doc: Any, data_set_ids: Iterable[int]) -> dict[str, int]:
    items = _result(doc)
    ids = set(data_set_ids)
    eds = {
        i["reporting_unit_summary"]["reporting_unit_code"]
        for i in items
        if i["data_set_id"] in ids
        and _is_hospital(i["reporting_unit_summary"])
        and (i.get("value") or 0) > 0
    }
    return {"data_items": len(items), "ed_hospitals_2023_24": len(eds)}


def _counter_for(source: Source) -> Counter:
    if source.id == "myhospitals_reporting_units":
        return count_reporting_units
    if source.id == "myhospitals_ed_presentations":
        ids = source.pin["data_set_ids_2023_24"]
        return lambda doc: count_ed_presentations(doc, ids)
    raise KeyError(f"no count check defined for volatile source {source.id!r}")


def check_expect(source: Source, counts: dict[str, int]) -> None:
    """Raise ManifestMismatch if any count moved by more than `tolerance_pct` from the manifest."""
    tol = float(source.expect.get("tolerance_pct", 0)) / 100
    problems = []
    for key, expected in source.expect.items():
        if key == "tolerance_pct":
            continue
        got = counts.get(key)
        if got is None:
            problems.append(f"{key}: not computed")
        elif abs(got - expected) > tol * expected:
            problems.append(f"{key}: got {got}, manifest says {expected}")
    if problems:
        raise ManifestMismatch(f"upstream changed: {source.id}: " + "; ".join(problems))


def _check_volatile(source: Source, path: Path) -> None:
    try:
        with path.open(encoding="utf-8-sig") as f:  # the API sends a byte-order mark
            doc = json.load(f)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ManifestMismatch(f"upstream changed: {source.id} is not valid JSON ({e})") from e
    check_expect(source, _counter_for(source)(doc))


def _quarantine(path: Path) -> Path:
    """Move a file that failed its check aside, so it can be inspected but is never used."""
    bad = path.with_name(path.name + ".mismatch")
    path.replace(bad)
    return bad


def fetch_source(source: Source, raw_dir: Path, base_dir: Path, force: bool = False) -> Path:
    dest = source.path_in(raw_dir)
    url = resolve_url(source.url, base_dir)
    accept = source.extra.get("accept")
    check = _check_volatile if source.volatile else verify
    if dest.exists() and not force:
        try:
            check(source, dest)
            log.info("%s already present and checked", source.id)
            return dest
        except ManifestMismatch:
            log.warning("%s on disk failed its check; downloading again", source.id)
    log.info("downloading %s%s", source.id, " (live API)" if source.volatile else "")
    download(url, dest, accept=accept)
    try:
        check(source, dest)
    except ManifestMismatch:
        log.error("%s failed its check; kept for inspection at %s", source.id, _quarantine(dest))
        raise
    return dest


def fetch_all(
    manifest: Manifest, raw_dir: Path, only: Iterable[str] | None = None, force: bool = False
) -> dict[str, Path]:
    wanted = list(only) if only else manifest.ids
    unknown = sorted(set(wanted) - set(manifest.ids))
    if unknown:
        raise KeyError(f"unknown source ids: {unknown}")
    base = manifest.path.parent
    return {sid: fetch_source(manifest.get(sid), raw_dir, base, force=force) for sid in wanted}
