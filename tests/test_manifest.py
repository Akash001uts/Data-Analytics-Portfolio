import shutil

import pytest
import yaml

from dap.common.manifest import (
    ManifestError,
    ManifestMismatch,
    load_manifest,
    resolve_url,
    sha256_file,
    verify,
)


def test_repo_manifest_is_complete(repo_manifest):
    m = load_manifest(repo_manifest)
    assert "aihw_pph_sa3" in m.ids and "phidu_sha_pha" in m.ids
    for s in m.sources:
        assert s.url.startswith("https://"), s.id
        assert s.licence, s.id
        if not s.volatile:
            assert s.bytes > 0, s.id
    assert len({s.file for s in m.sources}) == len(m.sources)


def test_target_source_is_all_hospitals(repo_manifest):
    target = load_manifest(repo_manifest).get("aihw_pph_sa3")
    assert target.extra["used_for"] == ["target"]
    assert "public and private" in target.extra["coverage"]


def _write(tmp_path, sources):
    p = tmp_path / "manifest.yaml"
    p.write_text(yaml.safe_dump({"sources": sources}), encoding="utf-8")
    return p


def test_pinned_source_needs_hash(tmp_path):
    p = _write(tmp_path, [{"id": "a", "url": "https://x", "file": "a", "licence": "x"}])
    with pytest.raises(ManifestError, match="sha256"):
        load_manifest(p)


def test_volatile_source_needs_expect(tmp_path):
    p = _write(
        tmp_path, [{"id": "a", "url": "https://x", "file": "a", "licence": "x", "volatile": True}]
    )
    with pytest.raises(ManifestError, match="expect"):
        load_manifest(p)


def test_duplicate_ids_rejected(tmp_path):
    entry = {
        "id": "a",
        "url": "https://x",
        "file": "a",
        "licence": "x",
        "volatile": True,
        "expect": {"n": 1},
    }
    with pytest.raises(ManifestError, match="duplicate"):
        load_manifest(_write(tmp_path, [entry, entry]))


def test_verify_passes_then_catches_a_changed_file(fixtures_dir, tmp_path):
    src = load_manifest(fixtures_dir / "manifest.yaml").get("fixture_target")
    copy = tmp_path / "target.csv"
    shutil.copy(fixtures_dir / "target.csv", copy)
    verify(src, copy)
    with copy.open("a", encoding="utf-8") as f:
        f.write("tampered\n")
    with pytest.raises(ManifestMismatch, match="upstream changed"):
        verify(src, copy)


def test_sha256_matches_known_value(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_relative_file_urls_resolve_against_manifest(fixtures_dir):
    url = resolve_url("file:target.csv", fixtures_dir)
    assert url.startswith("file:///") and url.endswith("/target.csv")
    assert resolve_url("https://example.org/a.xlsx", fixtures_dir) == "https://example.org/a.xlsx"
