"""Run the RoBERTa sentiment model once and cache its outputs, so nothing else needs torch.

The model was trained on tweets, not product reviews, and is used as is (zero-shot). Long
reviews are truncated to the model's 512-token limit instead of being dropped, which is what
my first version did. The cache holds review ids, the three class probabilities and the token
count before truncation: no review text.

Needs `uv sync --group nlp`.
"""

import csv
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd

from dap.common import paths
from dap.sentiment.clean import LABELS

log = logging.getLogger(__name__)

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
REVISION = "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"  # pinned commit on the Hugging Face Hub
MODEL_LICENCE = "CC BY 4.0"
MAX_LENGTH = 512
BATCH = 16
COLUMNS = ["review_id", "p_negative", "p_neutral", "p_positive", "n_tokens"]


def cache_path() -> Path:
    return paths.reports_dir() / "sentiment" / "roberta_predictions.csv"


def read_cache(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or cache_path())
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run `uv sync --group nlp` and `dap sentiment transformer` first"
        )
    return pd.read_csv(path).set_index("review_id")


def predict(reviews: pd.DataFrame, out: Path | None = None, limit: int | None = None) -> Path:
    """Score `reviews` (review_id, text) and append to the cache, skipping ids already in it.

    Reviews are sorted by length so each batch pads as little as possible.
    """
    os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out = Path(out or cache_path())
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set(pd.read_csv(out).review_id) if out.exists() else set()
    todo = reviews[~reviews.review_id.isin(done)]
    if limit:
        todo = todo.head(limit)
    if todo.empty:
        return out

    torch.manual_seed(0)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    # The pinned commit only has pytorch_model.bin, which is what gets loaded. transformers would
    # also start a background download of a safetensors copy from an unmerged bot pull request
    # ("for next time"); the environment variable above switches that off.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, revision=REVISION, use_safetensors=False
    ).eval()
    order = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    if sorted(order) != sorted(LABELS):
        raise RuntimeError(f"unexpected model labels {order}")

    n_tokens = [len(ids) for ids in tok(list(todo.text), truncation=False)["input_ids"]]
    todo = todo.assign(n_tokens=n_tokens).sort_values("n_tokens")
    new = not out.exists()
    start = time.perf_counter()
    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        if new:
            writer.writerow(COLUMNS)
        for i in range(0, len(todo), BATCH):
            batch = todo.iloc[i : i + BATCH]
            enc = tok(
                list(batch.text),
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            )
            with torch.inference_mode():
                probs = torch.softmax(model(**enc).logits, dim=-1).numpy()
            for rid, p, n in zip(batch.review_id, probs, batch.n_tokens, strict=True):
                by_label = dict(zip(order, p, strict=True))
                writer.writerow([rid, *(f"{by_label[lab]:.4f}" for lab in LABELS), n])
            f.flush()
            if (i // BATCH) % 50 == 0:
                rate = (i + len(batch)) / (time.perf_counter() - start)
                log.info("%d / %d reviews, %.1f per second", i + len(batch), len(todo), rate)

    meta = {
        "model": MODEL,
        "revision": REVISION,
        "licence": MODEL_LICENCE,
        "max_length": MAX_LENGTH,
        "label_order_in_model": order,
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
    }
    out.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out
