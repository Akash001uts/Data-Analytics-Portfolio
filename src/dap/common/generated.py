"""Generated blocks in Markdown files: text between two HTML comment markers, rewritten from a
results file. Each project README keeps its numbers in one of these, and a test checks they match.
"""

import re
from pathlib import Path


def _pattern(start: str, end: str) -> re.Pattern:
    return re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)


def current_block(path: Path, start: str, end: str) -> str:
    m = _pattern(start, end).search(Path(path).read_text(encoding="utf-8"))
    return m.group(0) if m else ""


def replace_block(path: Path, start: str, end: str, block: str) -> bool:
    """Swap the block in `path` for `block` (markers included). True if the file changed."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    pattern = _pattern(start, end)
    if not pattern.search(text):
        raise ValueError(f"{path} has no generated block starting {start!r}")
    new = pattern.sub(lambda _m: block, text)
    if new != text:
        path.write_text(new, encoding="utf-8", newline="\n")
    return new != text
