"""The models, from a national mean up to LightGBM, all behind one small interface.

Each model has `fit(data, train)` and `predict(data, test)`, where `data` holds every area and
`train` / `test` are integer positions. Passing positions rather than sliced frames lets the spatial
lag model see the whole map (it needs every area's features to spread predictions along the
weights), but nothing is ever fitted on a test area's target, imputed value or scaled value.

Every model predicts log(PPH rate). Fitting weights are the area populations, rescaled to mean 1
inside each training fold so they don't change the meaning of the regularisation settings.
"""

import contextlib
import io
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from libpysal.weights import W
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dap.common.seeds import SEED
from dap.health.features import check_columns
from dap.health.spatial import subset_weights

RA_CLASSES = ("Major cities", "Inner regional", "Outer regional", "Remote")
RA_SHARES = ("ra_inner_regional_share", "ra_outer_regional_share", "ra_remote_share")

# A compact, interpretable set for the spatial lag model. Maximum likelihood is unstable with all
# 42 overlapping features, so this keeps one or two per idea: disadvantage, age, Aboriginal
# population, remoteness, insurance, GP use and distance.
LAG_FEATURES = (
    "irsd_score",
    "age_65_plus_pct",
    "aboriginal_pct",
    "ra_outer_regional_share",
    "ra_remote_share",
    "private_health_insurance_pct",
    "gp_services_per_100",
    "km_to_public_hospital",
)

LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 8,
    "min_child_samples": 15,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "n_jobs": 1,
    "verbose": -1,
}


@dataclass
class ModelData:
    """Everything a model may use. `X` only ever holds allowlisted features."""

    X: pd.DataFrame
    y: pd.Series  # log(PPH age-standardised rate)
    weight: pd.Series  # population (ERP), used to weight fitting and metrics
    meta: pd.DataFrame  # sa4_code and state_code: grouping keys, never model inputs
    w: W | None = None  # spatial weights in the same order as X
    tiers: tuple[str, ...] = ("A",)
    names: pd.Series = field(default_factory=lambda: pd.Series(dtype=str))

    def __post_init__(self):
        check_columns(list(self.X.columns), self.tiers)
        for part in (self.y, self.weight, self.meta):
            if not part.index.equals(self.X.index):
                raise ValueError("X, y, weight and meta must share one index")
        if self.w is not None and list(self.w.id_order) != list(self.X.index):
            raise ValueError("the spatial weights are not in the same order as X")

    def __len__(self) -> int:
        return len(self.X)


def fold_weights(weight: pd.Series, idx: np.ndarray) -> np.ndarray:
    w = weight.to_numpy(dtype=float)[idx]
    return w / w.mean()


def ra_class(X: pd.DataFrame) -> pd.Series:
    """Each area's dominant remoteness class, from the ABS population shares."""
    shares = X[list(RA_SHARES)]
    full = pd.concat([1 - shares.sum(axis=1), shares], axis=1)
    full.columns = list(RA_CLASSES)
    return full.idxmax(axis=1).where(shares.notna().all(axis=1), "Unknown")


class NationalMean:
    name = "national_mean"
    label = "National mean"

    def fit(self, data: ModelData, train: np.ndarray):
        self.mean_ = float(
            np.average(data.y.to_numpy()[train], weights=fold_weights(data.weight, train))
        )
        return self

    def predict(self, data: ModelData, test: np.ndarray) -> np.ndarray:
        return np.full(len(test), self.mean_)


class RemotenessStateMeans:
    """Population-weighted mean of the log rate in each state x remoteness cell.

    A cell needs `min_areas` training areas; otherwise the area falls back to its remoteness class
    mean, then to the national mean. Remoteness comes from the ABS shares, not the AIHW peer groups
    (those come with the target and mix disadvantage into the city groups).
    """

    name = "remoteness_state"
    label = "Remoteness x state means"

    def __init__(self, min_areas: int = 3):
        self.min_areas = min_areas

    def fit(self, data: ModelData, train: np.ndarray):
        df = pd.DataFrame(
            {
                "y": data.y.to_numpy()[train],
                "w": fold_weights(data.weight, train),
                "state": data.meta.state_code.to_numpy()[train],
                "ra": ra_class(data.X).to_numpy()[train],
            }
        )

        def means(keys):
            g = df.groupby(keys)
            m = g.apply(lambda d: np.average(d.y, weights=d.w), include_groups=False)
            return m[g.size() >= self.min_areas].to_dict()

        self.cell_ = means(["state", "ra"])
        self.ra_ = means("ra")
        self.national_ = float(np.average(df.y, weights=df.w))
        return self

    def predict(self, data: ModelData, test: np.ndarray) -> np.ndarray:
        states = data.meta.state_code.to_numpy()[test]
        ras = ra_class(data.X).to_numpy()[test]
        return np.array(
            [
                self.cell_.get((s, r), self.ra_.get(r, self.national_))
                for s, r in zip(states, ras, strict=True)
            ]
        )


class Ridge:
    """Median imputation, standardising and ridge regression, all fitted on the training fold.

    The penalty is chosen by RidgeCV's efficient leave-one-out on the training rows only.
    """

    name = "ridge"
    label = "Ridge regression"

    def __init__(self, columns: tuple[str, ...] | None = None):
        self.columns = columns

    def _cols(self, data: ModelData) -> list[str]:
        return list(self.columns or data.X.columns)

    def fit(self, data: ModelData, train: np.ndarray):
        self.pipeline_ = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("ridge", RidgeCV(alphas=np.logspace(-2, 3, 31))),
            ]
        )
        X = data.X[self._cols(data)].iloc[train]
        self.pipeline_.fit(
            X, data.y.iloc[train], ridge__sample_weight=fold_weights(data.weight, train)
        )
        return self

    def predict(self, data: ModelData, test: np.ndarray) -> np.ndarray:
        return self.pipeline_.predict(data.X[self._cols(data)].iloc[test])

    def coefficients(self) -> pd.Series:
        names = self.pipeline_.named_steps["impute"].get_feature_names_out()
        return pd.Series(self.pipeline_.named_steps["ridge"].coef_, index=names)


class SpatialLag:
    """Spatial lag model, log(rate) = rho W log(rate) + X b + e, by maximum likelihood (spreg).

    Fitted on the training areas with the weights cut down to those areas. Predicting with the
    observed rates of neighbours would leak: under SA4-grouped CV most neighbours of a test area
    are in the test fold too. So predictions use the reduced form, (I - rho W)^-1 X b over the whole
    map, which needs features only. spreg's ML_Lag takes no case weights, so this model is fitted
    unweighted (its errors are still population-weighted, like the others).
    """

    name = "spatial_lag"
    label = "Spatial lag (ML)"

    def __init__(self, columns: tuple[str, ...] = LAG_FEATURES):
        self.columns = columns

    def _prep(self):
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])

    def fit(self, data: ModelData, train: np.ndarray):
        from spreg import ML_Lag

        if data.w is None:
            raise ValueError("the spatial lag model needs spatial weights")
        self.prep_ = self._prep()
        X = self.prep_.fit_transform(data.X[list(self.columns)].iloc[train])
        ids = list(data.X.index[train])
        w_train = subset_weights(data.w, ids)
        # spreg prints the model name on every fit
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            m = ML_Lag(data.y.iloc[train].to_numpy()[:, None], X, w_train, method="full")
        betas = m.betas.ravel()
        self.intercept_, self.coef_, self.rho_ = betas[0], betas[1:-1], float(betas[-1])
        self.model_ = m
        return self

    def predict(self, data: ModelData, test: np.ndarray) -> np.ndarray:
        X = self.prep_.transform(data.X[list(self.columns)])
        xb = self.intercept_ + X @ self.coef_
        wfull, _ = data.w.full()
        y_all = np.linalg.solve(np.eye(len(data)) - self.rho_ * wfull, xb)
        return y_all[test]


class LightGBM:
    """Gradient-boosted trees with fixed, conservative settings for about 260 training areas.

    The settings are not tuned on the outer folds. LightGBM handles missing values itself, so there
    is no imputer or scaler to leak.
    """

    name = "lightgbm"
    label = "LightGBM"

    def __init__(self, columns: tuple[str, ...] | None = None, **params):
        self.columns = columns
        self.params = {**LGBM_PARAMS, **params}

    def _cols(self, data: ModelData) -> list[str]:
        return list(self.columns or data.X.columns)

    def fit(self, data: ModelData, train: np.ndarray):
        from lightgbm import LGBMRegressor

        self.model_ = LGBMRegressor(**self.params)
        self.model_.fit(
            data.X[self._cols(data)].iloc[train],
            data.y.iloc[train],
            sample_weight=fold_weights(data.weight, train),
        )
        return self

    def predict(self, data: ModelData, test: np.ndarray) -> np.ndarray:
        return self.model_.predict(data.X[self._cols(data)].iloc[test])

    def shap_values(self, data: ModelData) -> pd.DataFrame:
        """Exact TreeSHAP contributions on the log scale (LightGBM's pred_contrib)."""
        X = data.X[self._cols(data)]
        contrib = self.model_.predict(X, pred_contrib=True)
        return pd.DataFrame(contrib[:, :-1], index=X.index, columns=X.columns)


class StateIntercept:
    """A base model plus a random intercept for each state, shrunk towards zero.

    The residuals of the feature-only models lean by state, so this learns how far each state sits
    from the base model. Training residuals would understate that (a flexible model partly fits
    them), so the residuals come from an inner SA4-grouped CV on the training rows only. Each
    state's offset is its population-weighted mean residual times n / (n + k), an empirical Bayes
    shrinkage where k = residual variance / between-state variance (method of moments). The base
    model is then refitted on every training row and the offset added to its predictions.

    A state with no training areas gets no offset. Under SA4 folds that happens to the ACT (one SA4)
    whenever it's held out, so the ACT is always predicted from features alone.
    """

    base: type = Ridge
    inner_splits = 5

    def _inner_residuals(self, data: ModelData, train: np.ndarray) -> np.ndarray:
        from sklearn.model_selection import GroupKFold

        resid = np.full(len(train), np.nan)
        groups = data.meta.sa4_code.to_numpy()[train]
        n = min(self.inner_splits, len(set(groups)))
        cv = GroupKFold(n_splits=n, shuffle=True, random_state=SEED)
        for tr, te in cv.split(train, groups=groups):
            pred = self.base().fit(data, train[tr]).predict(data, train[te])
            resid[te] = data.y.to_numpy()[train[te]] - pred
        return resid

    def fit(self, data: ModelData, train: np.ndarray):
        resid = self._inner_residuals(data, train)
        df = pd.DataFrame(
            {
                "r": resid,
                "w": fold_weights(data.weight, train),
                "state": data.meta.state_code.to_numpy()[train],
            }
        )
        g = df.groupby("state")
        mean = g.apply(lambda d: np.average(d.r, weights=d.w), include_groups=False)
        # Effective sample size of a weighted mean, so big-city areas don't count as many states
        n_eff = g.w.sum() ** 2 / g.w.apply(lambda w: (w**2).sum())
        within = float(np.average((df.r - df.state.map(mean)) ** 2, weights=df.w))
        between = float(np.var(mean.to_numpy(), ddof=1)) if len(mean) > 1 else 0.0
        tau2 = max(between - np.mean(within / n_eff), 0.0)
        self.k_ = within / tau2 if tau2 > 0 else np.inf
        self.state_sd_ = float(np.sqrt(tau2))
        self.offset_ = (mean * n_eff / (n_eff + self.k_)).to_dict()
        self.model_ = self.base().fit(data, train)
        return self

    def predict(self, data: ModelData, test: np.ndarray) -> np.ndarray:
        states = data.meta.state_code.to_numpy()[test]
        offset = np.array([self.offset_.get(s, 0.0) for s in states])
        return self.model_.predict(data, test) + offset


class RidgeState(StateIntercept):
    name = "ridge_state"
    label = "Ridge + state intercept"
    base = Ridge


class LightGBMState(StateIntercept):
    name = "lightgbm_state"
    label = "LightGBM + state intercept"
    base = LightGBM


MODELS = (
    NationalMean,
    RemotenessStateMeans,
    Ridge,
    SpatialLag,
    LightGBM,
    RidgeState,
    LightGBMState,
)
# Feature-only models: the residual map uses the best of these, so a state-wide lean stays visible
FEATURE_MODELS = ("ridge", "spatial_lag", "lightgbm")
