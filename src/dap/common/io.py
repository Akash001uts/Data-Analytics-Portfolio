"""Small file helpers. JSON is written sorted and atomically so results.json diffs stay readable."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)
