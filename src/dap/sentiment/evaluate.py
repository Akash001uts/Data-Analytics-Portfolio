"""Metrics, paired bootstrap confidence intervals and error slices.

The bootstrap is paired: every model is scored on the same resampled reviews in each round, so
the interval on the difference between two models accounts for them being tested on the same
reviews. "Model A is better" needs that interval to sit clear of zero.
"""

import re
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from dap.common.seeds import SEED
from dap.sentiment.clean import LABELS

N_BOOT = 2000
LENGTH_BINS = [0, 50, 100, 200, np.inf]
LENGTH_LABELS = ["up to 50 words", "51 to 100", "101 to 200", "over 200"]
# Reviews that state their own rating ("I'd give it three stars") hand a word model the label
STARS = re.compile(r"\bstars?\b", re.IGNORECASE)
# A crude, labelled proxy for "the review says two things", not a measure of mixed sentiment
CONTRAST = re.compile(r"\b(?:but|however|although|though|except|unfortunately)\b", re.IGNORECASE)


def _codes(labels) -> np.ndarray:
    lookup = {lab: i for i, lab in enumerate(LABELS)}
    return np.array([lookup[x] for x in labels], dtype=np.int64)


def macro_f1_codes(y: np.ndarray, p: np.ndarray) -> float:
    k = len(LABELS)
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    return float(f1.mean())


def scores(y, pred) -> dict:
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y, pred, labels=LABELS)
    return {
        "macro_f1": float(f.mean()),
        "accuracy": float((np.asarray(y) == np.asarray(pred)).mean()),
        "per_class": {
            lab: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "n": int(n[i]),
            }
            for i, lab in enumerate(LABELS)
        },
        "confusion": cm.tolist(),  # rows: true label, columns: predicted, in LABELS order
    }


def paired_bootstrap(y, preds: dict[str, np.ndarray], n_boot: int = N_BOOT, seed: int = SEED):
    """Macro-F1 for every model on the same resamples, plus every pairwise difference."""
    yc = _codes(y)
    pc = {m: _codes(p) for m, p in preds.items()}
    rng = np.random.default_rng(seed)
    n = len(yc)
    draws = {m: np.empty(n_boot) for m in preds}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        for m in preds:
            draws[m][b] = macro_f1_codes(yc[idx], pc[m][idx])

    def interval(estimate, values):
        lo, hi = np.percentile(values, [2.5, 97.5])
        return {"estimate": float(estimate), "ci_low": float(lo), "ci_high": float(hi)}

    single = {m: interval(macro_f1_codes(yc, pc[m]), draws[m]) for m in preds}
    diffs = {}
    for a, b in combinations(preds, 2):
        est = single[a]["estimate"] - single[b]["estimate"]
        diffs[f"{a} minus {b}"] = interval(est, draws[a] - draws[b])
    return single, diffs


def slices(df: pd.DataFrame, preds: dict[str, np.ndarray]) -> dict:
    """Macro-F1 and size of each slice, for each model. `df` needs text, label and n_tokens."""
    words = df.text.str.split().str.len()
    groups = {
        "length": pd.cut(words, LENGTH_BINS, labels=LENGTH_LABELS).astype(str),
        "truncated_by_roberta": np.where(df.n_tokens > 512, "truncated", "fits in 512 tokens"),
        # The fine-tuned model only reads the first 128 tokens (same tokeniser as RoBERTa)
        "cut_for_finetuned": np.where(df.n_tokens > 128, "cut", "fits in 128 tokens"),
        "contrast_word": np.where(df.text.str.contains(CONTRAST), "has a contrast word", "none"),
        "mentions_stars": np.where(df.text.str.contains(STARS), "mentions stars", "no mention"),
    }
    out = {}
    y = df.label.to_numpy()
    for gname, g in groups.items():
        g = np.asarray(g)
        out[gname] = {}
        for value in pd.unique(g):
            mask = g == value
            out[gname][str(value)] = {"n": int(mask.sum())} | {
                m: macro_f1_codes(_codes(y[mask]), _codes(p[mask])) for m, p in preds.items()
            }
    return out
