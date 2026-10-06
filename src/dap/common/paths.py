"""Repository paths. Set DAP_ROOT to run against a different checkout or a temp directory."""

import os
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("DAP_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find the repository root; set DAP_ROOT")


def data_dir() -> Path:
    return repo_root() / "data"


def raw_dir() -> Path:
    return data_dir() / "raw"


def interim_dir() -> Path:
    return data_dir() / "interim"


def processed_dir() -> Path:
    return data_dir() / "processed"


def reports_dir() -> Path:
    return repo_root() / "reports"


def manifest_path() -> Path:
    return data_dir() / "manifest.yaml"
