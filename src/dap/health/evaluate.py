"""Cross-validation and metrics.

Two schemes with the same number of folds: random K-fold, and K-fold grouped by SA4 so that no SA4
is split between training and test. Neighbouring SA3s are alike, so a random split lets a model
lean on a test area's neighbours; the gap between the two schemes measures how much.

Models predict log(rate). Errors are reported on the original scale (admissions per 100,000) after
exp(), which estimates the median rate for an area like this one rather than the mean. Every metric
is weighted by population.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from dap.common.seeds import SEED
from dap.health.models import ModelData

N_SPLITS = 5
N_REPEATS = 5
SCHEMES = ("spatial", "random")


def splits(
    data: ModelData, scheme: str, repeat: int = 0, n_splits: int = N_SPLITS
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    seed = SEED + repeat
    if scheme == "spatial":
        cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yield from cv.split(data.X, groups=data.meta.sa4_code)
    elif scheme == "random":
        yield from KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(data.X)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")


def weighted_metrics(y_true, y_pred, weight) -> dict:
    """Population-weighted MAE, RMSE and R2 (R2 against the weighted mean of the true values)."""
    y_true, y_pred, w = (np.asarray(a, dtype=float) for a in (y_true, y_pred, weight))
    err = y_true - y_pred
    mean = np.average(y_true, weights=w)
    return {
        "mae": float(np.average(np.abs(err), weights=w)),
        "rmse": float(np.sqrt(np.average(err**2, weights=w))),
        "r2": float(
            1 - np.average(err**2, weights=w) / np.average((y_true - mean) ** 2, weights=w)
        ),
    }


def score(data: ModelData, log_pred: np.ndarray) -> dict:
    out = weighted_metrics(np.exp(data.y), np.exp(log_pred), data.weight)
    out["r2_log"] = weighted_metrics(data.y, log_pred, data.weight)["r2"]
    return out


@dataclass
class CVResult:
    oof: pd.DataFrame  # one column of out-of-fold log predictions per repeat
    fitted: list  # (repeat, fold, train, test, fitted model), kept for checks and tests
    metrics: pd.DataFrame  # one row per repeat


def run_cv(
    make_model: Callable[[], object],
    data: ModelData,
    scheme: str,
    n_repeats: int = N_REPEATS,
    n_splits: int = N_SPLITS,
    keep_fitted: bool = False,
) -> CVResult:
    oof = pd.DataFrame(index=data.X.index, dtype=float)
    fitted, rows = [], []
    for r in range(n_repeats):
        pred = np.full(len(data), np.nan)
        for k, (train, test) in enumerate(splits(data, scheme, r, n_splits)):
            model = make_model().fit(data, train)
            pred[test] = model.predict(data, test)
            if keep_fitted:
                fitted.append((r, k, train, test, model))
        if np.isnan(pred).any():
            raise RuntimeError("some areas were never in a test fold")
        oof[r] = pred
        rows.append(score(data, pred))
    return CVResult(oof=oof, fitted=fitted, metrics=pd.DataFrame(rows))


def summarise(metrics: pd.DataFrame) -> dict:
    return {m: {"mean": float(metrics[m].mean()), "sd": float(metrics[m].std())} for m in metrics}
