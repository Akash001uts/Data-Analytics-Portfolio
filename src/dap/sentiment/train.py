"""`dap sentiment train`: fit the baselines, score every model on one sample, write results.

RoBERTa's and the fine-tuned model's predictions come from committed caches (`dap sentiment
transformer` and `dap sentiment finetune` make them), so this step needs no torch.
Everything the README quotes comes from `reports/sentiment/results.json`.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from dap.common import paths
from dap.common.io import sig, write_json
from dap.common.seeds import SEED, set_seed
from dap.sentiment import evaluate as E
from dap.sentiment import finetune
from dap.sentiment import models as M
from dap.sentiment.clean import LABELS
from dap.sentiment.data import load_reviews
from dap.sentiment.transformer import cache_path, read_cache

log = logging.getLogger(__name__)

LABELS_FOR = {
    "vader_default": "VADER, default thresholds",
    "vader_tuned": "VADER, thresholds tuned on validation",
    "tfidf_logistic": "TF-IDF + logistic regression",
    "roberta": "RoBERTa (Twitter sentiment, zero-shot)",
    "finetuned": "DistilRoBERTa, fine-tuned on training reviews",
}
VADER_TUNING_SIZE = 20_000  # validation reviews used to pick VADER's thresholds


def predictions_path() -> Path:
    return paths.processed_dir() / "sentiment_eval_predictions.pkl"


def reports_dir() -> Path:
    return paths.reports_dir() / "sentiment"


def with_roberta(ev: pd.DataFrame) -> pd.DataFrame:
    rob = read_cache()
    missing = set(ev.review_id) - set(rob.index)
    if missing:
        raise FileNotFoundError(
            f"{cache_path()} has no predictions for {len(missing)} evaluation reviews; "
            "run `dap sentiment transformer`"
        )
    return ev.join(rob, on="review_id")


def with_finetuned(ev: pd.DataFrame) -> pd.DataFrame:
    """The fine-tuned model's probabilities, as ft_p_negative and so on."""
    ft = finetune.read_cache()
    missing = set(ev.review_id) - set(ft.index)
    if missing:
        raise FileNotFoundError(
            f"{finetune.cache_path()} has no predictions for {len(missing)} evaluation reviews; "
            "run `dap sentiment finetune`"
        )
    probs = ft[[f"p_{lab}" for lab in LABELS]].add_prefix("ft_")
    return ev.join(probs, on="review_id")


def predict_all(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict, M.TfidfLogistic]:
    train = df[df.split == "train"]
    val = df[df.split == "validation"]
    ev = with_finetuned(with_roberta(df[df.in_eval].sort_values("review_id")))

    log.info("VADER: tuning thresholds on %d validation reviews", VADER_TUNING_SIZE)
    tune = val.sample(min(VADER_TUNING_SIZE, len(val)), random_state=SEED)
    thresholds, vader_val_f1 = M.tune_vader(M.vader_compound(tune.text), tune.label.to_numpy())
    compound = M.vader_compound(ev.text)

    log.info("TF-IDF: fitting on %d training reviews", len(train))
    tfidf = M.TfidfLogistic().fit(train.text, train.label, val.text, val.label)

    preds = {
        "vader_default": M.vader_labels(compound, M.VADER_DEFAULT),
        "vader_tuned": M.vader_labels(compound, thresholds),
        "tfidf_logistic": tfidf.predict(ev.text),
        "roberta": M.roberta_labels(ev),
        "finetuned": M.roberta_labels(ev, prefix="ft_p_"),
    }
    settings = {
        "vader_default_thresholds": list(M.VADER_DEFAULT),
        "vader_tuned_thresholds": list(thresholds),
        "vader_tuning_reviews": len(tune),
        "vader_tuned_validation_macro_f1": vader_val_f1,
        "tfidf_params": tfidf.params_,
        "tfidf_vocabulary": len(tfidf.vectoriser.vocabulary_),
        "tfidf_validation_macro_f1": tfidf.validation_macro_f1_,
        "tfidf_search": tfidf.search_,
    }
    ev = ev.assign(vader_compound=compound, **{f"pred_{m}": p for m, p in preds.items()})
    return ev, preds, settings, tfidf


def train(out_dir: Path | None = None) -> dict:
    from dap.sentiment import figures, readme

    set_seed()
    df, counts = load_reviews()
    ev, preds, settings, tfidf = predict_all(df)
    y = ev.label.to_numpy()

    single, diffs = E.paired_bootstrap(y, preds)
    models = {}
    for m, p in preds.items():
        models[m] = {"label": LABELS_FOR[m], **E.scores(y, p), "macro_f1": single[m]}
    meta = json.loads(cache_path().with_suffix(".json").read_text(encoding="utf-8"))
    ft_path = finetune.cache_path().with_suffix(".json")
    ft_meta = json.loads(ft_path.read_text(encoding="utf-8"))
    results = {
        "data": counts
        | {
            "eval_label_share": ev.label.value_counts(normalize=True).reindex(LABELS).to_dict(),
            "text_used": "review text only (not the summary line)",
            "labels": "1-2 stars negative, 3 neutral, 4-5 positive",
            "split": "by reviewer, 70/10/20 train/validation/test; evaluation sample from test",
        },
        "settings": settings
        | {"roberta": meta, "finetuned": ft_meta, "bootstrap_resamples": E.N_BOOT, "seed": SEED},
        "models": models,
        "differences": diffs,
        "slices": E.slices(ev, preds),
        "tfidf_top_terms": tfidf.top_terms(),
    }
    results = sig(results)
    out = Path(out_dir or reports_dir())
    results_path = out / "results.json"
    write_json(results_path, results)
    if out_dir is None:
        readme.update(results)
    # Per-review predictions (with text) stay local, for the notebook's error examples
    keep = ["review_id", "label", "score", "text", "n_tokens", "vader_compound"]
    ev[keep + [f"pred_{m}" for m in preds]].to_pickle(predictions_path())
    figs = figures.make_all(results, out / "figures")
    return {"results": results_path, "figures": figs, "stats": results, "eval": ev}
