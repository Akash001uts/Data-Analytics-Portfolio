"""Leakage and correctness checks for the models and cross-validation, on a synthetic lattice.

The lattice is 8 x 8 square "SA3s" in 2 x 2 blocks that act as SA4s, split into two "states".
The target is smooth in space, so neighbouring areas are alike, as in the real data.
"""

import numpy as np
import pandas as pd
import pytest
from libpysal.weights import lat2W

from dap.common.seeds import set_seed
from dap.health import evaluate as E
from dap.health.features import LeakageError
from dap.health.models import (
    LAG_FEATURES,
    MODELS,
    LightGBM,
    LightGBMState,
    ModelData,
    RemotenessStateMeans,
    Ridge,
    RidgeState,
    ra_class,
)
from dap.health.spatial import global_moran
from dap.health.train import sig

SIDE = 8
FEATURES = [*LAG_FEATURES, "ra_inner_regional_share", "unemployed_pct"]


def lattice(side: int = SIDE, seed: int = 0) -> ModelData:
    rng = np.random.default_rng(seed)
    r, c = np.divmod(np.arange(side * side), side)
    ids = [f"9{i:04d}" for i in range(side * side)]
    smooth = np.sin(r / 2.5) + np.cos(c / 3.0)
    X = pd.DataFrame(
        {f: smooth * rng.uniform(0.5, 2) + rng.normal(0, 0.5, side * side) for f in FEATURES},
        index=ids,
    )
    shares = rng.dirichlet([3, 1, 1, 1], side * side)
    X["ra_inner_regional_share"] = shares[:, 1]
    X["ra_outer_regional_share"] = shares[:, 2]
    X["ra_remote_share"] = shares[:, 3]
    X.iloc[::9, 1] = np.nan  # some missing values for the imputer
    y = pd.Series(8 + 0.3 * smooth + rng.normal(0, 0.1, side * side), index=ids)
    meta = pd.DataFrame(
        {
            "sa4_code": [f"{rr // 2}-{cc // 2}" for rr, cc in zip(r, c, strict=True)],
            "state_code": np.where(c < side // 2, "1", "2"),
        },
        index=ids,
    )
    w = lat2W(side, side, rook=False, id_type="string")
    w.remap_ids(ids)
    w.transform = "r"
    weight = pd.Series(rng.integers(5_000, 200_000, side * side).astype(float), index=ids)
    return ModelData(X=X, y=y, weight=weight, meta=meta, w=w)


@pytest.fixture(scope="module")
def data() -> ModelData:
    set_seed()
    return lattice()


# --- splits -----------------------------------------------------------------------------------


@pytest.mark.parametrize("repeat", range(3))
def test_spatial_folds_never_split_an_sa4(data, repeat):
    tested = []
    for train, test in E.splits(data, "spatial", repeat):
        sa4 = data.meta.sa4_code.to_numpy()
        assert not set(sa4[train]) & set(sa4[test])
        assert not set(train) & set(test)
        tested.extend(test)
    assert sorted(tested) == list(range(len(data)))


def test_random_folds_do_share_sa4s(data):
    """Otherwise the spatial vs random comparison would mean nothing."""
    sa4 = data.meta.sa4_code.to_numpy()
    shared = [set(sa4[tr]) & set(sa4[te]) for tr, te in E.splits(data, "random")]
    assert all(shared)


def test_unknown_scheme_is_rejected(data):
    with pytest.raises(ValueError):
        next(E.splits(data, "by_vibes"))


# --- leakage ----------------------------------------------------------------------------------


def test_imputer_and_scaler_are_fitted_on_training_rows_only(data):
    """Plant huge values in every area so any test row in the fit would move the statistics."""
    X = data.X.copy()
    for train, test in E.splits(data, "spatial"):
        planted = X.copy()
        planted.iloc[test] = 1e6
        d = ModelData(X=planted, y=data.y, weight=data.weight, meta=data.meta, w=data.w)
        model = Ridge().fit(d, train)
        imputer = model.pipeline_.named_steps["impute"]
        scaler = model.pipeline_.named_steps["scale"]
        tr = X.iloc[train]
        np.testing.assert_allclose(imputer.statistics_, tr.median().to_numpy())
        np.testing.assert_allclose(scaler.mean_, tr.fillna(tr.median()).mean().to_numpy())
        assert scaler.mean_.max() < 1e3


@pytest.mark.parametrize("cls", MODELS, ids=lambda c: c.name)
def test_test_targets_never_reach_the_model(data, cls):
    """Change the test areas' targets: the test predictions must not move. This catches a spatial
    lag fed with neighbours' observed rates, or baselines averaged over all rows."""
    for train, test in list(E.splits(data, "spatial"))[:2]:
        before = cls().fit(data, train).predict(data, test)
        y = data.y.copy()
        y.iloc[test] = y.iloc[test] * 3 + 5
        moved = ModelData(X=data.X, y=y, weight=data.weight, meta=data.meta, w=data.w)
        after = cls().fit(moved, train).predict(moved, test)
        np.testing.assert_allclose(before, after)


@pytest.mark.parametrize("column", ["pph_asr", "erp", "aihw_sa3_group", "pph_chronic_asr"])
def test_model_data_rejects_columns_off_the_allowlist(data, column):
    X = data.X.assign(**{column: 1.0})
    with pytest.raises(LeakageError):
        ModelData(X=X, y=data.y, weight=data.weight, meta=data.meta)


def test_tier_b_needs_the_sensitivity_flag(data):
    X = data.X.assign(diabetes_asr=1.0)
    with pytest.raises(LeakageError):
        ModelData(X=X, y=data.y, weight=data.weight, meta=data.meta)
    ModelData(X=X, y=data.y, weight=data.weight, meta=data.meta, tiers=("A", "B"))


def test_misaligned_weights_are_rejected(data):
    X = data.X.iloc[::-1]
    with pytest.raises(ValueError):
        ModelData(
            X=X,
            y=data.y.iloc[::-1],
            weight=data.weight.iloc[::-1],
            meta=data.meta.iloc[::-1],
            w=data.w,
        )


# --- models and metrics -----------------------------------------------------------------------


def test_cv_beats_the_national_mean_on_a_learnable_target(data):
    base = E.run_cv(MODELS[0], data, "spatial", n_repeats=1).metrics.r2_log[0]
    ridge = E.run_cv(Ridge, data, "spatial", n_repeats=1).metrics.r2_log[0]
    assert base < 0.05 < ridge


def test_cv_is_repeatable(data):
    a = E.run_cv(LightGBM, data, "spatial", n_repeats=2).oof
    b = E.run_cv(LightGBM, data, "spatial", n_repeats=2).oof
    pd.testing.assert_frame_equal(a, b)


def test_remoteness_state_falls_back_when_a_cell_is_thin(data):
    model = RemotenessStateMeans(min_areas=10_000).fit(data, np.arange(len(data)))
    assert model.cell_ == {} and model.ra_ == {}
    np.testing.assert_allclose(model.predict(data, np.arange(3)), model.national_)


def shifted(data: ModelData, shift: float) -> ModelData:
    """The lattice with state "2" lifted by `shift` on the log scale."""
    y = data.y + np.where(data.meta.state_code == "2", shift, 0.0)
    return ModelData(X=data.X, y=y, weight=data.weight, meta=data.meta, w=data.w)


def test_state_intercept_recovers_a_planted_state_shift(data):
    d = shifted(data, 0.5)
    model = RidgeState().fit(d, np.arange(len(d)))
    gap = model.offset_["2"] - model.offset_["1"]
    assert 0.2 < gap <= 0.5 + 1e-9  # shrunk towards zero, never past the truth
    plain = E.run_cv(Ridge, d, "spatial", n_repeats=1).metrics.r2_log[0]
    with_state = E.run_cv(RidgeState, d, "spatial", n_repeats=1).metrics.r2_log[0]
    assert with_state > plain


def test_state_intercept_is_zero_for_an_unseen_state(data):
    """Like the ACT when its only SA4 is held out: no training areas, so no offset."""
    d = shifted(data, 0.5)
    train = np.flatnonzero(d.meta.state_code == "1")
    test = np.flatnonzero(d.meta.state_code == "2")
    model = LightGBMState().fit(d, train)
    assert "2" not in model.offset_
    np.testing.assert_allclose(model.predict(d, test), model.model_.predict(d, test))


def test_state_intercept_inner_residuals_use_training_rows_only(data):
    """Every inner residual must come from a model that never saw that row's target."""
    train = np.arange(0, len(data), 2)
    resid = RidgeState()._inner_residuals(data, train)
    assert not np.isnan(resid).any()
    in_sample = data.y.to_numpy()[train] - Ridge().fit(data, train).predict(data, train)
    assert np.mean(resid**2) > np.mean(in_sample**2)


def test_ra_class_picks_the_largest_share():
    X = pd.DataFrame(
        {
            "ra_inner_regional_share": [0.1, 0.6, 0.0, np.nan],
            "ra_outer_regional_share": [0.1, 0.2, 0.1, 0.5],
            "ra_remote_share": [0.0, 0.1, 0.8, 0.5],
        }
    )
    assert list(ra_class(X)) == ["Major cities", "Inner regional", "Remote", "Unknown"]


def test_weighted_metrics_by_hand():
    m = E.weighted_metrics([1, 2, 3], [1, 2, 5], [1, 1, 2])
    assert m["mae"] == pytest.approx(1.0)  # (0 + 0 + 2 * 2) / 4
    assert m["rmse"] == pytest.approx(np.sqrt(2.0))  # (2 * 4) / 4
    # weighted mean 2.25; total sum of squares (1.5625 + 0.0625 + 2 * 0.5625) / 4 = 0.6875
    assert m["r2"] == pytest.approx(1 - 2.0 / 0.6875)
    assert E.weighted_metrics([1, 2, 3], [1, 2, 3], [1, 1, 1])["r2"] == 1.0


def test_moran_finds_the_smooth_target_but_not_noise(data):
    assert global_moran(data.y, data.w)["I"] > 0.3
    noise = pd.Series(np.random.default_rng(1).normal(size=len(data)), index=data.X.index)
    assert abs(global_moran(noise, data.w)["I"]) < 0.2


def test_sig_rounds_nested_values():
    assert sig({"a": [123456.0, 0.000123456], "b": np.float64(2 / 3), "c": "x"}) == {
        "a": [123500.0, 0.0001235],
        "b": 0.6667,
        "c": "x",
    }
