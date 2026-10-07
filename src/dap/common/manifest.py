"""Data manifest: where each raw file comes from, and a check that it is the file I analysed.

Pinned sources are checked by size and SHA256. Volatile sources (live APIs) have no stable hash, so
they are checked against the record counts in their `expect` block instead (see dap.health.fetch).
"""

import hashlib
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from dap.common.io import read_yaml

USER_AGENT = "Mozilla/5.0 (compatible; dap-fetch/0.1; +https://github.com/Akash001uts/Data-Analytics-Portfolio)"
CHUNK = 1 << 20
_HEX = set("0123456789abcdef")


class ManifestError(ValueError):
    """The manifest itself is malformed."""


class ManifestMismatch(RuntimeError):
    """A downloaded file does not match the manifest, usually because the upstream file changed."""


@dataclass(frozen=True)
class Source:
    id: str
    url: str
    file: str
    licence: str
    title: str = ""
    sha256: str | None = None
    bytes: int | None = None
    volatile: bool = False
    expect: dict[str, Any] = field(default_factory=dict)
    pin: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def path_in(self, raw_dir: Path) -> Path:
        return Path(raw_dir) / self.id / self.file


_KNOWN = {"id", "url", "file", "licence", "title", "sha256", "bytes", "volatile", "expect", "pin"}


def _parse_source(entry: dict[str, Any]) -> Source:
    missing = [k for k in ("id", "url", "file", "licence") if not entry.get(k)]
    if missing:
        raise ManifestError(f"source {entry.get('id', '?')!r} is missing {missing}")
    volatile = bool(entry.get("volatile", False))
    sha = entry.get("sha256")
    if not volatile:
        if not (isinstance(sha, str) and len(sha) == 64 and set(sha) <= _HEX):
            raise ManifestError(f"source {entry['id']!r} needs a lowercase 64-character sha256")
        if not isinstance(entry.get("bytes"), int):
            raise ManifestError(f"source {entry['id']!r} needs an integer byte count")
    elif not entry.get("expect"):
        raise ManifestError(f"volatile source {entry['id']!r} needs an expect block")
    return Source(
        id=entry["id"],
        url=entry["url"],
        file=entry["file"],
        licence=entry["licence"],
        title=entry.get("title", ""),
        sha256=sha,
        bytes=entry.get("bytes"),
        volatile=volatile,
        expect=dict(entry.get("expect") or {}),
        pin=dict(entry.get("pin") or {}),
        extra={k: v for k, v in entry.items() if k not in _KNOWN},
    )


@dataclass(frozen=True)
class Manifest:
    path: Path
    sources: tuple[Source, ...]

    def get(self, source_id: str) -> Source:
        for s in self.sources:
            if s.id == source_id:
                return s
        raise KeyError(f"no source {source_id!r} in {self.path}")

    @property
    def ids(self) -> list[str]:
        return [s.id for s in self.sources]


def load_manifest(path: Path) -> Manifest:
    path = Path(path)
    doc = read_yaml(path)
    if not isinstance(doc, dict) or not isinstance(doc.get("sources"), list):
        raise ManifestError(f"{path} must contain a 'sources' list")
    sources = tuple(_parse_source(e) for e in doc["sources"])
    ids = [s.id for s in sources]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ManifestError(f"duplicate source ids: {dupes}")
    return Manifest(path=path, sources=sources)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def verify(source: Source, path: Path) -> None:
    """Raise ManifestMismatch unless `path` has the size and hash the manifest records."""
    if source.volatile:
        raise ValueError(f"{source.id} is volatile; check it with its expect block instead")
    path = Path(path)
    size = path.stat().st_size
    if size != source.bytes:
        raise ManifestMismatch(
            f"upstream changed: {source.id} is {size} bytes, manifest says {source.bytes} "
            f"({source.url})"
        )
    digest = sha256_file(path)
    if digest != source.sha256:
        raise ManifestMismatch(
            f"upstream changed: {source.id} sha256 is {digest}, manifest says {source.sha256} "
            f"({source.url})"
        )


def resolve_url(url: str, base_dir: Path) -> str:
    """Allow `file:relative/path` in fixture manifests, resolved against the manifest's folder."""
    parsed = urlparse(url)
    if parsed.scheme == "file" and not url.startswith("file://"):
        return (Path(base_dir) / unquote(parsed.path)).resolve().as_uri()
    return url


def download(url: str, dest: Path, accept: str | None = None, timeout: float = 300) -> Path:
    """Stream `url` to `dest` via a temp file in its folder, so a failed download leaves nothing."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(url, headers=headers)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(req, timeout=timeout) as resp:
            shutil.copyfileobj(resp, out, CHUNK)
        Path(tmp).replace(dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return dest
