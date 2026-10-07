"""The README's results block must match reports/results.json exactly. Needs no data."""

from dap.common import paths
from dap.common.io import read_json
from dap.health import readme


def test_readme_block_matches_results_json():
    results = read_json(paths.reports_dir() / "results.json")
    assert readme.current_block() == readme.render(results), (
        "README results are stale: run `uv run dap health train`"
    )


def test_update_replaces_only_the_block(tmp_path):
    results = read_json(paths.reports_dir() / "results.json")
    page = tmp_path / "README.md"
    page.write_text(
        f"before\n\n{readme.START}\nold numbers\n{readme.END}\n\nafter\n", encoding="utf-8"
    )
    assert readme.update(results, page)
    text = page.read_text(encoding="utf-8")
    assert text.startswith("before\n\n") and text.endswith("\n\nafter\n")
    assert "old numbers" not in text
    assert not readme.update(results, page)  # a second run changes nothing


def test_sentiment_readme_block_matches_results_json():
    from dap.sentiment import readme as sreadme

    results = read_json(paths.reports_dir() / "sentiment" / "results.json")
    assert sreadme.current_block() == sreadme.render(results), (
        "sentiment README results are stale: run `uv run dap sentiment train`"
    )
