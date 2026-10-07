"""Fine-tune a small transformer on the training reviewers and cache its predictions.

RoBERTa zero-shot was trained on tweets and never saw a product review, so this checks how much
of the gap to TF-IDF comes from that. It runs on a CPU, so it's deliberately small:
DistilRoBERTa, a stratified sample of the training reviews, reviews cut to their first 128
tokens, one epoch. The checkpoint with the best macro-F1 on a sample of the validation reviewers
is kept; the evaluation sample is only scored once, at the end, like every other model.

The cache holds review ids, the three class probabilities and the token count: no review text.
The weights stay local in data/processed. Needs `uv sync --group nlp`.
"""

import csv
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dap.common import paths
from dap.common.seeds import SEED
from dap.sentiment.clean import LABELS

log = logging.getLogger(__name__)

MODEL = "distilbert/distilroberta-base"
REVISION = "fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b"  # pinned commit on the Hugging Face Hub
MODEL_LICENCE = "Apache 2.0"
MAX_LENGTH = 128
TRAIN_SIZE = 20_000  # training reviews used (stratified), out of about 275,000
VAL_SIZE = 2_000  # validation reviews used to pick the checkpoint
BATCH = 16
EPOCHS = 1
CHECKS_PER_EPOCH = 4
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_SHARE = 0.06
COLUMNS = ["review_id", "p_negative", "p_neutral", "p_positive", "n_tokens"]


def cache_path() -> Path:
    return paths.reports_dir() / "sentiment" / "finetuned_predictions.csv"


def model_dir() -> Path:
    return paths.processed_dir() / "sentiment_finetuned"


def read_cache(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or cache_path())
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run `uv sync --group nlp` and `dap sentiment finetune` first"
        )
    return pd.read_csv(path).set_index("review_id")


def stratified_sample(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """`n` rows with the same label shares as `df`, independent of row order."""
    df = df.sort_values("review_id")
    if len(df) <= n:
        return df
    shares = df.label.value_counts(normalize=True)
    take = (shares * n).round().astype(int)
    take[take.idxmax()] += n - take.sum()
    parts = [df[df.label == lab].sample(k, random_state=seed) for lab, k in take.items() if k > 0]
    return pd.concat(parts).sort_values("review_id")


def training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The fine-tuning sample from the training reviewers and the checkpoint sample from validation.

    Nothing from the test reviewers, so the evaluation sample stays untouched.
    """
    train = stratified_sample(df[df.split == "train"], TRAIN_SIZE)
    val = stratified_sample(df[df.split == "validation"], VAL_SIZE)
    return train, val


def class_weights(labels) -> np.ndarray:
    """'Balanced' weights in LABELS order, so the rare neutral class isn't ignored."""
    counts = pd.Series(labels).value_counts().reindex(LABELS, fill_value=0).to_numpy()
    return len(labels) / (len(LABELS) * np.maximum(counts, 1))


def length_batches(lengths, batch: int, seed: int) -> list[np.ndarray]:
    """Shuffled batches of similar-length reviews, so each batch pads as little as possible.

    Indices are shuffled, cut into chunks of 50 batches, sorted by length inside each chunk, and
    the batches themselves shuffled. Every index appears exactly once.
    """
    rng = np.random.default_rng(seed)
    lengths = np.asarray(lengths)
    idx = rng.permutation(len(lengths))
    chunk = batch * 50
    out = []
    for i in range(0, len(idx), chunk):
        part = idx[i : i + chunk]
        part = part[np.argsort(lengths[part], kind="stable")]
        out += [part[j : j + batch] for j in range(0, len(part), batch)]
    order = rng.permutation(len(out))
    return [out[i] for i in order]


def _probs(model, tok, texts: list[str], batch: int = 64) -> np.ndarray:
    import torch

    lengths = np.array([len(t) for t in texts])
    order = np.argsort(lengths, kind="stable")
    probs = np.empty((len(texts), len(LABELS)), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(order), batch):
            sel = order[i : i + batch]
            enc = tok(
                [texts[j] for j in sel],
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            )
            probs[sel] = torch.softmax(model(**enc).logits, dim=-1).numpy()
    return probs


def fit(df: pd.DataFrame, limit: int | None = None) -> dict:
    """Fine-tune on the training sample and save the best checkpoint to `model_dir()`."""
    os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    from dap.sentiment.evaluate import macro_f1_codes

    torch.manual_seed(SEED)
    train, val = training_data(df)
    if limit:
        train = stratified_sample(train, limit)
    codes = {lab: i for i, lab in enumerate(LABELS)}
    y_train = train.label.map(codes).to_numpy()
    y_val = val.label.map(codes).to_numpy()

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL,
        revision=REVISION,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=codes,
    )
    enc = tok(list(train.text), truncation=True, max_length=MAX_LENGTH)["input_ids"]
    batches = [
        b
        for epoch in range(EPOCHS)
        for b in length_batches([len(x) for x in enc], BATCH, SEED + epoch)
    ]
    check_every = max(1, len(batches) // (EPOCHS * CHECKS_PER_EPOCH))
    decay = [
        p for n, p in model.named_parameters() if not n.endswith("bias") and "LayerNorm" not in n
    ]
    no_decay = [p for n, p in model.named_parameters() if n.endswith("bias") or "LayerNorm" in n]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=LEARNING_RATE,
    )
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP_SHARE * len(batches)), len(batches))
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights(train.label), dtype=torch.float32)
    )

    history, best = [], None
    start = time.perf_counter()
    for step, b in enumerate(batches, start=1):
        model.train()
        padded = tok.pad({"input_ids": [enc[i] for i in b]}, return_tensors="pt")
        loss = loss_fn(model(**padded).logits, torch.tensor(y_train[b]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        if step % 50 == 0:
            rate = step * BATCH / (time.perf_counter() - start)
            log.info(
                "step %d / %d, loss %.3f, %.1f reviews per second",
                step,
                len(batches),
                loss.item(),
                rate,
            )
        if step % check_every == 0 or step == len(batches):
            pred = _probs(model, tok, list(val.text)).argmax(axis=1)
            f1 = macro_f1_codes(y_val, pred)
            history.append({"step": step, "reviews_seen": step * BATCH, "validation_macro_f1": f1})
            log.info("step %d: validation macro-F1 %.3f", step, f1)
            if best is None or f1 > best["validation_macro_f1"]:
                best = history[-1]
                model.save_pretrained(model_dir())
                tok.save_pretrained(model_dir())

    return {
        "model": MODEL,
        "revision": REVISION,
        "licence": MODEL_LICENCE,
        "max_length": MAX_LENGTH,
        "training_reviews": len(train),
        "training_label_share": train.label.value_counts(normalize=True).reindex(LABELS).to_dict(),
        "validation_reviews": len(val),
        "batch_size": BATCH,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_share": WARMUP_SHARE,
        "loss": "cross-entropy with balanced class weights",
        "class_weights": dict(zip(LABELS, class_weights(train.label).tolist(), strict=True)),
        "history": history,
        "best_step": best["step"],
        "validation_macro_f1": best["validation_macro_f1"],
        "training_minutes": round((time.perf_counter() - start) / 60, 1),
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
    }


def predict(reviews: pd.DataFrame, meta: dict, out: Path | None = None) -> Path:
    """Score `reviews` (review_id, text) with the saved checkpoint and write the cache."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out = Path(out or cache_path())
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(model_dir())
    model = AutoModelForSequenceClassification.from_pretrained(model_dir())
    order = [model.config.id2label[i] for i in range(model.config.num_labels)]
    if order != list(LABELS):
        raise RuntimeError(f"unexpected model labels {order}")
    reviews = reviews.sort_values("review_id")
    texts = list(reviews.text)
    start = time.perf_counter()
    probs = _probs(model, tok, texts)
    n_tokens = [len(ids) for ids in tok(texts, truncation=False)["input_ids"]]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(COLUMNS)
        for rid, p, n in zip(reviews.review_id, probs, n_tokens, strict=True):
            writer.writerow([rid, *(f"{x:.4f}" for x in p), n])
    meta = meta | {"prediction_minutes": round((time.perf_counter() - start) / 60, 1)}
    out.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out


def run(df: pd.DataFrame, limit: int | None = None) -> Path:
    meta = fit(df, limit=limit)
    return predict(df[df.in_eval][["review_id", "text"]], meta)
