"""The baselines: VADER (rule-based) and TF-IDF with logistic regression.

Anything that gets tuned (VADER's thresholds, the regression's C and class weights) is tuned on the
validation reviewers only. The evaluation sample is touched once, at the end.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from dap.sentiment.clean import LABELS

VADER_DEFAULT = (-0.05, 0.05)  # the thresholds from the VADER paper, and what my first version used
THRESHOLD_GRID = np.round(np.arange(-0.95, 0.96, 0.05), 2)
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


def roberta_labels(probs: pd.DataFrame, prefix: str = "p_") -> np.ndarray:
    cols = [f"{prefix}{lab}" for lab in LABELS]
    return np.array(LABELS)[probs[cols].to_numpy().argmax(axis=1)]
