"""Find the verified raw reviews file and load it (parse, clean, dedupe, split)."""

from pathlib import Path

import pandas as pd

from dap.common import paths
from dap.common.manifest import load_manifest, verify
from dap.sentiment.clean import load

SOURCE_ID = "snap_finefoods"


def raw_path(raw_dir: Path | None = None) -> Path:
    source = load_manifest(paths.sentiment_manifest_path()).get(SOURCE_ID)
    path = source.path_in(Path(raw_dir or paths.raw_dir()))
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `dap sentiment fetch` first")
    verify(source, path)  # never analyse a file that has drifted from the manifest
    return path


def load_reviews(raw_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    return load(raw_path(raw_dir))
