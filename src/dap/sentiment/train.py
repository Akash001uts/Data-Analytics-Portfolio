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
from sklearn.metrics import f1_score

from dap.common import paths
from dap.common.io import sig, write_json
from dap.common.seeds import SEED, set_seed
from dap.sentiment import evaluate as E
from dap.sentiment import finetune
from dap.sentiment import models as M
from dap.sentiment.clean import LABELS
from dap.sentiment.data import load_reviews
from dap.sentiment.transformer import cache_path, read_cache, tuning_sample, validation_cache_path

log = logging.getLogger(__name__)

LABELS_FOR = {
    "vader_default": "VADER, default thresholds",
    "vader_tuned": "VADER, thresholds tuned on validation",
    "tfidf_logistic": "TF-IDF + logistic regression",
    "roberta": "RoBERTa (Twitter sentiment, zero-shot)",
    "roberta_tuned": "RoBERTa, decision rule tuned on validation",
    "roberta_calibrated": "RoBERTa, recalibrated on validation (logistic regression)",
    "finetuned": "DistilRoBERTa, fine-tuned on training reviews",
}
VADER_TUNING_SIZE = 20_000  # validation reviews used to pick VADER's thresholds


def predictions_path() -> Path:
    return paths.processed_dir() / "sentiment_eval_predictions.pkl"


def reports_dir() -> Path:
    return paths.reports_dir() / "sentiment"


def with_roberta(ev: pd.DataFrame, path: Path | None = None) -> pd.DataFrame:
    path = Path(path or cache_path())
    rob = read_cache(path)
    missing = set(ev.review_id) - set(rob.index)
    if missing:
        split = " --split validation" if path == validation_cache_path() else ""
        raise FileNotFoundError(
            f"{path} has no predictions for {len(missing)} of the reviews it needs; "
            f"run `dap sentiment transformer{split}`"
        )
    return ev.join(rob, on="review_id")


def roberta_rule(df: pd.DataFrame, path: Path | None = None) -> tuple[dict[str, float], dict]:
    """Tune RoBERTa's log-probability offsets on the scored validation sample (no test rows)."""
    tune = with_roberta(tuning_sample(df), path or validation_cache_path())
    log.info("RoBERTa: tuning the decision rule on %d validation reviews", len(tune))
    offsets, tuned_f1 = M.tune_roberta(tune, tune.label.to_numpy())
    argmax_f1 = f1_score(tune.label, M.roberta_labels(tune), average="macro", labels=LABELS)
    grid = M.OFFSET_GRID
    return offsets, {
        "roberta_tuned_log_offsets": offsets,
        "roberta_offset_grid": {
            "low": float(grid.min()),
            "high": float(grid.max()),
            "step": float(round(grid[1] - grid[0], 6)),
        },
        "roberta_offsets_inside_grid": M.offsets_inside_grid(offsets),
        "roberta_tuning_reviews": len(tune),
        "roberta_argmax_validation_macro_f1": float(argmax_f1),
        "roberta_tuned_validation_macro_f1": tuned_f1,
    }


def roberta_calibration(
    df: pd.DataFrame, path: Path | None = None
) -> tuple[M.RobertaCalibration, dict]:
    """Fit the regression on RoBERTa's scores, on the same validation sample as the offsets."""
    tune = with_roberta(tuning_sample(df), path or validation_cache_path())
    log.info("RoBERTa: recalibrating on %d validation reviews", len(tune))
    cal = M.RobertaCalibration().fit(tune, tune.label.to_numpy())
    return cal, {
        "roberta_calibrated_class_weight": str(cal.class_weight_),
        "roberta_calibrated_search": cal.search_,
        "roberta_calibrated_cv_macro_f1": cal.cv_macro_f1_,
        "roberta_calibrated_coefficients": cal.coefficients(),
    }


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

    offsets, rule_settings = roberta_rule(df)
    calibration, calibration_settings = roberta_calibration(df)

    log.info("TF-IDF: fitting on %d training reviews", len(train))
    tfidf = M.TfidfLogistic().fit(train.text, train.label, val.text, val.label)

    preds = {
        "vader_default": M.vader_labels(compound, M.VADER_DEFAULT),
        "vader_tuned": M.vader_labels(compound, thresholds),
        "tfidf_logistic": tfidf.predict(ev.text),
        "roberta": M.roberta_labels(ev),
        "roberta_tuned": M.roberta_labels(ev, offsets),
        "roberta_calibrated": calibration.predict(ev),
        "finetuned": M.roberta_labels(ev, prefix="ft_p_"),
    }
    settings = (
        {
            "vader_default_thresholds": list(M.VADER_DEFAULT),
            "vader_tuned_thresholds": list(thresholds),
            "vader_tuning_reviews": len(tune),
            "vader_tuned_validation_macro_f1": vader_val_f1,
            "tfidf_params": tfidf.params_,
            "tfidf_vocabulary": len(tfidf.vectoriser.vocabulary_),
            "tfidf_validation_macro_f1": tfidf.validation_macro_f1_,
            "tfidf_search": tfidf.search_,
        }
        | rule_settings
        | calibration_settings
    )
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
