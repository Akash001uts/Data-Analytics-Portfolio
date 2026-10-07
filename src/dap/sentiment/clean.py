"""Read the SNAP Fine Food Reviews file, clean the text, dedupe, and split by reviewer.

The order matters for leakage. Duplicates are removed before any split, and splits are by
reviewer, so no review text and no reviewer appears in more than one of train, validation and
test. A reviewer's way of writing would otherwise let a word-based model recognise them.
"""

import gzip
import hashlib
import html
import re
from pathlib import Path

import pandas as pd

from dap.common.seeds import SEED

FIELDS = {
    b"product/productId": "product_id",
    b"review/userId": "user_id",
    b"review/profileName": "profile_name",
    b"review/helpfulness": "helpfulness",
    b"review/score": "score",
    b"review/time": "time",
    b"review/summary": "summary",
    b"review/text": "text",
}
LABELS = ("negative", "neutral", "positive")
SPLITS = {"train": 70, "validation": 10, "test": 20}  # % of reviewers in each split
EVAL_SIZE = 10_000

_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_SPACE = re.compile(r"\s+")


def _decode(raw: bytes) -> str:
    """The file mixes encodings: most lines are UTF-8, a few are Windows-1252 or Latin-1."""
    for enc in ("utf-8", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("latin-1")


def parse(path: Path) -> pd.DataFrame:
    """One row per review. `review_id` is the review's position in the pinned file (from 0).

    A line that doesn't start with a known field name continues the previous field (some profile
    names contain line breaks).
    """
    rows, cur, last = [], {}, None
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as f:
        for line in f:
            line = line.rstrip(b"\r\n")
            if not line.strip():
                if cur:
                    rows.append(cur)
                    cur, last = {}, None
                continue
            key, sep, value = line.partition(b": ")
            if sep and key in FIELDS:
                last = FIELDS[key]
                cur[last] = _decode(value)
            elif last is not None:
                cur[last] += " " + _decode(line)
    if cur:
        rows.append(cur)
    df = pd.DataFrame(rows, columns=list(FIELDS.values()))
    df.insert(0, "review_id", range(len(df)))
    df["score"] = df.score.astype(float).astype(int)
    df["time"] = df.time.astype(int)
    return df


def clean_text(s: str) -> str:
    return _SPACE.sub(" ", html.unescape(_BR.sub(" ", s))).strip()


def normalise(s: str) -> str:
    """The key used to spot duplicate texts: lower case, single spaces."""
    return _SPACE.sub(" ", s.lower()).strip()


def to_label(score: pd.Series) -> pd.Series:
    """1 to 2 stars negative, 3 neutral, 4 to 5 positive."""
    return pd.cut(score, [0, 2, 3, 5], labels=list(LABELS)).astype(str)


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[["review_id", "product_id", "user_id", "score", "time", "summary", "text"]].copy()
    df["text"] = df.text.map(clean_text)
    df["label"] = to_label(df.score)
    df["norm"] = df.text.map(normalise)
    return df


def dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop repeated reviews before any split.

    1. The same reviewer and text under several products (Amazon shows a review on every variant
       of a product): keep the first copy.
    2. The same text from different reviewers: keep the first copy if they agree on the label,
       and drop every copy if they don't, because there's no right answer for that text.
    """
    df = df.sort_values("review_id")
    counts = {"reviews": len(df)}
    step1 = df.drop_duplicates(["user_id", "norm"])
    counts["dropped_same_reviewer_same_text"] = len(df) - len(step1)
    labels_per_text = step1.groupby("norm").label.transform("nunique")
    conflicting = labels_per_text > 1
    counts["dropped_cross_reviewer_conflicting_label"] = int(conflicting.sum())
    step2 = step1[~conflicting]
    out = step2.drop_duplicates("norm")
    counts["dropped_cross_reviewer_same_label"] = len(step2) - len(out)
    counts["after_dedupe"] = len(out)
    return out.reset_index(drop=True), counts


def reviewer_split(user_id: pd.Series, seed: int = SEED) -> pd.Series:
    """Assign each reviewer to train, validation or test by a hash of their id.

    hashlib, not Python's hash(), which changes from one run to the next.
    """
    edges, total = [], 0
    for name, pct in SPLITS.items():
        total += pct
        edges.append((total, name))

    def pick(uid: str) -> str:
        bucket = int(hashlib.sha256(f"{seed}:{uid}".encode()).hexdigest(), 16) % 100
        return next(name for edge, name in edges if bucket < edge)

    return user_id.map(pick)


def eval_sample(test: pd.DataFrame, n: int = EVAL_SIZE, seed: int = SEED) -> pd.DataFrame:
    """A fixed, stratified sample of the test reviewers' reviews (label shares kept)."""
    test = test.sort_values("review_id")  # so the sample never depends on row order
    if len(test) <= n:
        return test
    shares = test.label.value_counts(normalize=True)
    take = (shares * n).round().astype(int)
    take[take.idxmax()] += n - take.sum()
    parts = [
        test[test.label == lab].sample(k, random_state=seed) for lab, k in take.items() if k > 0
    ]
    return pd.concat(parts).sort_values("review_id")


def load(path: Path) -> tuple[pd.DataFrame, dict]:
    """Parse, clean, dedupe and split. Returns every kept review with a `split` column."""
    raw = parse(path)
    df, counts = dedupe(prepare(raw))
    df["split"] = reviewer_split(df.user_id)
    test = df[df.split == "test"]
    df["in_eval"] = df.review_id.isin(eval_sample(test).review_id)
    counts.update(
        {f"{s}_reviews": int((df.split == s).sum()) for s in SPLITS}
        | {f"{s}_reviewers": int(df[df.split == s].user_id.nunique()) for s in SPLITS}
        | {"eval_sample": int(df.in_eval.sum())}
    )
    return df, counts
