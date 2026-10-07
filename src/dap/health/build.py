"""`dap health build`: raw files in, one cleaned SA3 GeoPackage and a data summary out."""

import logging
from pathlib import Path

from dap.common import paths
from dap.common.io import write_json
from dap.common.manifest import load_manifest, verify
from dap.health.clean import build_sa3_table, summarise

log = logging.getLogger(__name__)

TABLE_NAME = "health_sa3.gpkg"
SUMMARY_NAME = "data_summary.json"


def build(
    raw_dir: Path | None = None, out_dir: Path | None = None, reports_dir: Path | None = None
) -> dict:
    manifest = load_manifest(paths.manifest_path())
    raw_dir = Path(raw_dir or paths.raw_dir())
    sources = {}
    for s in manifest.sources:
        path = s.path_in(raw_dir)
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run `dap health fetch` first")
        if not s.volatile:
            verify(s, path)  # never build from a file that has drifted from the manifest
        sources[s.id] = path
    ed_ids = manifest.get("myhospitals_ed_presentations").pin["data_set_ids_2023_24"]

    table = build_sa3_table(sources, ed_ids)
    out_dir = Path(out_dir or paths.processed_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / TABLE_NAME
    table.reset_index().to_file(table_path, layer="sa3", driver="GPKG")
    stats = summarise(table)
    summary_path = Path(reports_dir or paths.reports_dir()) / SUMMARY_NAME
    write_json(summary_path, stats)
    return {"table": table_path, "summary": summary_path, "stats": stats}
