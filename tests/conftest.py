from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def repo_manifest() -> Path:
    return REPO / "data" / "manifest.yaml"
