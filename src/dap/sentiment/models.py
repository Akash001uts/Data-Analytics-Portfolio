"""The baselines: VADER (rule-based) and TF-IDF with logistic regression.

Anything that gets tuned (VADER's thresholds, the regression's C and class weights, RoBERTa's
decision rule and its recalibration) is tuned on the validation reviewers only. The evaluation
sample is touched once, at the end.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from dap.common.seeds import SEED
from dap.sentiment.clean import LABELS

VADER_DEFAULT = (-0.05, 0.05)  # the thresholds from the VADER paper, and what my first version used
THRESHOLD_GRID = np.round(np.arange(-0.95, 0.96, 0.05), 2)
# Offsets added to RoBERTa's log probabilities for negative and neutral (positive stays at 0)
OFFSET_GRID = np.round(np.arange(-3.0, 3.01, 0.1), 1)
CALIBRATION_GRID = (None, "balanced")  # class weighting for the regression on RoBERTa's scores
TFIDF_GRID = [{"C": c, "class_weight": cw} for c in (0.5, 2.0, 8.0) for cw in (None, "balanced")]


def vader_compound(texts: pd.Series) -> np.ndarray:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    return np.array([sia.polarity_scores(t)["compound"] for t in texts])


def vader_labels(compound: np.ndarray, thresholds: tuple[float, float]) -> np.ndarray:
    lo, hi = thresholds
    return np.where(compound <= lo, "negative", np.where(compound >= hi, "positive", "neutral"))


def tune_vader(compound: np.ndarray, y: np.ndarray) -> tuple[tuple[float, float], float]:
    """The (low, high) thresholds with the best macro-F1 on the data given (validation only)."""
    best, best_f1 = VADER_DEFAULT, -1.0
    for i, lo in enumerate(THRESHOLD_GRID):
        for hi in THRESHOLD_GRID[i:]:
            f1 = f1_score(y, vader_labels(compound, (lo, hi)), average="macro", labels=LABELS)
            if f1 > best_f1:
                best, best_f1 = (float(lo), float(hi)), f1
    return best, best_f1


class TfidfLogistic:
    """Word unigrams and bigrams, then a multinomial logistic regression.

    The vocabulary and IDF weights are learnt from the training reviewers only.
    """

    name = "tfidf_logistic"
    label = "TF-IDF + logistic regression"

    def __init__(self, min_df: int = 3, max_features: int = 300_000):
        self.vectoriser = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=0.9,
            sublinear_tf=True,
            max_features=max_features,
            dtype=np.float32,
        )

    def fit(self, train_text, y_train, val_text, y_val, grid=TFIDF_GRID):
        X_train = self.vectoriser.fit_transform(train_text)
        X_val = self.vectoriser.transform(val_text)
        self.search_ = []
        best = None
        for params in grid:
            model = LogisticRegression(max_iter=2000, **params).fit(X_train, y_train)
            f1 = f1_score(y_val, model.predict(X_val), average="macro", labels=LABELS)
            self.search_.append({**params, "validation_macro_f1": float(f1)})
            if best is None or f1 > best[0]:
                best = (f1, params, model)
        self.validation_macro_f1_, self.params_, self.model_ = best
        return self

    def predict(self, text) -> np.ndarray:
        return self.model_.predict(self.vectoriser.transform(text))

    def top_terms(self, n: int = 15) -> dict[str, list[str]]:
        names = self.vectoriser.get_feature_names_out()
        out = {}
        for i, cls in enumerate(self.model_.classes_):
            out[str(cls)] = [str(t) for t in names[np.argsort(self.model_.coef_[i])[::-1][:n]]]
        return out


def roberta_labels(
    probs: pd.DataFrame, offsets: dict[str, float] | None = None, prefix: str = "p_"
) -> np.ndarray:
    """The most likely class, after adding an offset to each class's log probability.

    With no offsets this is plain argmax. An offset of +1 on neutral means "pick neutral even when
    it is up to e (about 2.7) times less likely than the top class".
    """
    cols = [f"{prefix}{lab}" for lab in LABELS]
    logp = np.log(np.clip(probs[cols].to_numpy(dtype=float), 1e-6, None))
    if offsets:
        logp = logp + np.array([offsets.get(lab, 0.0) for lab in LABELS])
    return np.array(LABELS)[logp.argmax(axis=1)]


def tune_roberta(probs: pd.DataFrame, y: np.ndarray) -> tuple[dict[str, float], float]:
    """The negative and neutral offsets with the best macro-F1 on the data given (validation only).

    Ties go to the smallest change from plain argmax, so a flat stretch of the grid can't drift.
    """
    best, best_key = None, None
    for neg in OFFSET_GRID:
        for neu in OFFSET_GRID:
            offsets = {"negative": float(neg), "neutral": float(neu), "positive": 0.0}
            pred = roberta_labels(probs, offsets)
            f1 = f1_score(y, pred, average="macro", labels=LABELS, zero_division=0)
            key = (round(f1, 10), -(abs(neg) + abs(neu)))
            if best_key is None or key > best_key:
                best, best_key = offsets, key
    return best, float(best_key[0])


def offsets_inside_grid(offsets: dict[str, float]) -> bool:
    """True when no tuned offset sits on the grid's edge (an edge means the grid was too small)."""
    lo, hi = float(OFFSET_GRID.min()), float(OFFSET_GRID.max())
    return all(lo < offsets[lab] < hi for lab in ("negative", "neutral"))


class RobertaCalibration:
    """A multinomial logistic regression on RoBERTa's three log probabilities ("matrix scaling").

    The offsets above can only shift each class's score; this can also reweight them against each
    other. It is fitted on validation reviews only, and the class weighting is picked by 5-fold
    cross-validation inside those reviews, so the fit never scores itself on rows it learnt from.
    """

    def fit(self, probs: pd.DataFrame, y: np.ndarray, prefix: str = "p_") -> "RobertaCalibration":
        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        x = self._features(probs, prefix)
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        self.search_ = {}
        for cw in CALIBRATION_GRID:
            pred = cross_val_predict(self._model(cw), x, y, cv=folds)
            f1 = f1_score(y, pred, average="macro", labels=LABELS, zero_division=0)
            self.search_[str(cw)] = float(f1)
        self.class_weight_ = max(CALIBRATION_GRID, key=lambda cw: self.search_[str(cw)])
        self.cv_macro_f1_ = self.search_[str(self.class_weight_)]
        self.model_ = self._model(self.class_weight_).fit(x, y)
        return self

    def predict(self, probs: pd.DataFrame, prefix: str = "p_") -> np.ndarray:
        return self.model_.predict(self._features(probs, prefix))

    def coefficients(self) -> dict[str, dict[str, float]]:
        """One row per predicted class: its weight on each log probability, and its intercept."""
        classes = self.model_.classes_
        if len(classes) == 2:  # a binary fit keeps one row, for the second class
            classes = classes[1:]
        out = {}
        for i, cls in enumerate(classes):
            weights = zip(LABELS, self.model_.coef_[i], strict=True)
            row = {f"log_p_{lab}": float(w) for lab, w in weights}
            out[str(cls)] = row | {"intercept": float(self.model_.intercept_[i])}
        return out

    @staticmethod
    def _model(class_weight):
        return LogisticRegression(class_weight=class_weight, max_iter=1000)

    @staticmethod
    def _features(probs: pd.DataFrame, prefix: str) -> np.ndarray:
        cols = [f"{prefix}{lab}" for lab in LABELS]
        return np.log(np.clip(probs[cols].to_numpy(dtype=float), 1e-6, None))
