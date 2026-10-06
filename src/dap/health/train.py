"""`dap health train`: spatial statistics, every model under both CV schemes, and results.json.

Everything the README and the notebooks quote comes from `reports/results.json`, which this writes.
Per-area predictions and residuals go to `data/processed/` (gitignored), for the maps.
"""

import logging
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from dap.common import paths
from dap.common.io import write_json
from dap.common.seeds import SEED, set_seed
from dap.health import readme, spatial
from dap.health.build import TABLE_NAME
from dap.health.evaluate import N_REPEATS, N_SPLITS, SCHEMES, run_cv, score, summarise
from dap.health.features import feature_names
from dap.health.models import (
    LAG_FEATURES,
    MODELS,
    LightGBM,
    ModelData,
    Ridge,
    SpatialLag,
)

log = logging.getLogger(__name__)

RESULTS_NAME = "results.json"
RESIDUALS_NAME = "health_sa3_results.gpkg"
ACT = "8"
TOP_N = 10


def load_table(path: Path | None = None) -> gpd.GeoDataFrame:
    path = Path(path or paths.processed_dir() / TABLE_NAME)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `dap health build` first")
    return gpd.read_file(path, layer="sa3").set_index("sa3_code")


def model_data(
    table: gpd.GeoDataFrame,
    target: str = "pph_asr",
    tiers: tuple[str, ...] = ("A",),
    exclude: tuple[str, ...] = (),
    drop_states: tuple[str, ...] = (),
    with_weights: bool = True,
) -> ModelData:
    """Rows with a published target and a population, the allowlisted features, and the weights."""
    rows = table[table[target].notna() & table.erp.notna() & ~table.state_code.isin(drop_states)]
    return ModelData(
        X=rows[feature_names(tiers, exclude)],
        y=np.log(rows[target]),
        weight=rows.erp,
        meta=rows[["sa4_code", "state_code"]],
        w=spatial.queen_weights(rows[["geometry"]]) if with_weights else None,
        tiers=tiers,
        names=rows.sa3_name,
    )


def sig(x, digits: int = 4):
    """Round floats to a few significant figures so results.json diffs stay quiet."""
    if isinstance(x, dict):
        return {k: sig(v, digits) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [sig(v, digits) for v in x]
    if isinstance(x, float | np.floating):
        x = float(x)
        if x == 0 or not math.isfinite(x):
            return x
        return round(x, digits - 1 - math.floor(math.log10(abs(x))))
    if isinstance(x, np.integer):
        return int(x)
    return x


def autocorrelation(data: ModelData) -> tuple[dict, pd.DataFrame]:
    lisa = spatial.local_moran(data.y, data.w)
    out = {
        "weights": spatial.describe_weights(data.w),
        "moran_log_pph": spatial.global_moran(data.y, data.w),
        "lisa_clusters": spatial.cluster_counts(lisa),
    }
    return out, lisa


def compare_models(data: ModelData) -> tuple[dict, dict[str, pd.Series]]:
    """Every model under spatial and random CV. Returns metrics and mean out-of-fold log predictions
    from the spatial scheme."""
    results, oof = {}, {}
    for cls in MODELS:
        entry = {"label": cls.label}
        for scheme in SCHEMES:
            log.info("%s, %s CV", cls.name, scheme)
            cv = run_cv(cls, data, scheme)
            entry[scheme] = summarise(cv.metrics)
            if scheme == "spatial":
                oof[cls.name] = cv.oof.mean(axis=1)
                # The single largest miss in either direction, across all repeats
                ratio = np.exp(cv.oof.sub(data.y, axis=0))
                for key, area in (
                    ("largest_over", ratio.max(axis=1).idxmax()),
                    ("largest_under", ratio.min(axis=1).idxmin()),
                ):
                    entry[key] = {
                        "sa3": data.names[area],
                        "observed": float(np.exp(data.y[area])),
                        "predicted": float(
                            np.exp(data.y[area])
                            * (
                                ratio.loc[area].max()
                                if key == "largest_over"
                                else ratio.loc[area].min()
                            )
                        ),
                    }
        entry["optimism_r2"] = entry["random"]["r2"]["mean"] - entry["spatial"]["r2"]["mean"]
        resid = data.y - oof[cls.name]
        entry["residual_moran_I"] = spatial.global_moran(resid, data.w)["I"]
        results[cls.name] = entry
    return results, oof


def sensitivity(table: gpd.GeoDataFrame) -> dict:
    """Ridge and LightGBM under spatial CV with other feature sets and targets."""
    variants = {
        "main": {},
        "without_private_insurance": {"exclude": ("private_health_insurance_pct",)},
        "with_tier_b_health_status": {"tiers": ("A", "B")},
        "chronic_conditions_target": {"target": "pph_chronic_asr"},
        "acute_conditions_target": {"target": "pph_acute_asr"},
        "target_2018_19_without_act": {"target": "pph_asr_2018_19", "drop_states": (ACT,)},
    }
    out = {}
    for name, kw in variants.items():
        data = model_data(table, with_weights=False, **kw)
        out[name] = {"areas": len(data)}
        for cls in (Ridge, LightGBM):
            log.info("sensitivity %s, %s", name, cls.name)
            m = summarise(run_cv(cls, data, "spatial").metrics)
            out[name][cls.name] = {"r2": m["r2"], "r2_log": m["r2_log"], "mae": m["mae"]}
    return out


def explain(data: ModelData) -> dict:
    """Full-data fits, for interpretation only (all reported errors come from CV)."""
    all_rows = np.arange(len(data))

    ridge = Ridge().fit(data, all_rows)
    coef = ridge.coefficients()
    coef = coef.reindex(coef.abs().sort_values(ascending=False).index)

    lag = SpatialLag().fit(data, all_rows)
    lag_resid = pd.Series(lag.model_.u.ravel(), index=data.X.index)

    lgbm = LightGBM().fit(data, all_rows)
    shap = lgbm.shap_values(data)
    mean_abs = shap.abs().mean().sort_values(ascending=False)
    # Rank correlation between a feature and its SHAP value: +1 means "higher value, higher rate".
    direction = {}
    for f in mean_abs.index[:TOP_N]:
        ok = data.X[f].notna()
        direction[f] = float(np.corrcoef(data.X[f][ok].rank(), shap[f][ok].rank())[0, 1])
    return {
        "ridge_standardised_coefficients": coef.head(15).to_dict(),
        "ridge_alpha": float(ridge.pipeline_.named_steps["ridge"].alpha_),
        "spatial_lag": {
            "features": list(LAG_FEATURES),
            "rho": lag.rho_,
            "standardised_coefficients": dict(zip(LAG_FEATURES, lag.coef_.tolist(), strict=True)),
            # Scored like the CV (population-weighted R2 on the log scale). spreg's own prediction
            # adds rho x the neighbours' observed rates; the reduced form uses features only, as
            # the cross-validated predictions do.
            "r2_log_in_sample_with_neighbour_rates": score(data, lag.model_.predy.ravel())[
                "r2_log"
            ],
            "r2_log_in_sample_features_only": score(data, lag.predict(data, all_rows))["r2_log"],
            "residual_moran_I": spatial.global_moran(lag_resid, data.w)["I"],
        },
        "shap_mean_abs_log": mean_abs.head(15).to_dict(),
        "shap_direction_rank_r": direction,
    }, shap


def residuals(
    data: ModelData, log_pred: pd.Series, model: str, states: pd.Series
) -> tuple[dict, pd.DataFrame]:
    """Observed / expected from out-of-fold (spatial CV) predictions, and LISA on the log ratio.

    `states` holds each area's state name. No model sees the state, so a state-wide lean in the
    residuals points at something state-level the features miss (or at how hospitals record).
    """
    log_ratio = data.y - log_pred
    lisa = spatial.local_moran(log_ratio, data.w)
    df = pd.DataFrame(
        {
            "sa3_name": data.names,
            "state": states.reindex(data.X.index),
            "observed": np.exp(data.y),
            "expected": np.exp(log_pred),
            "log_ratio": log_ratio,
            "pct_vs_expected": 100 * (np.exp(log_ratio) - 1),
            "resid_cluster": lisa.cluster,
        }
    )

    def top(frame):
        return [
            {
                "sa3": r.sa3_name,
                "state": r.state,
                "observed": r.observed,
                "expected": r.expected,
                "pct_vs_expected": r.pct_vs_expected,
            }
            for r in frame.itertuples()
        ]

    by = df.sort_values("log_ratio")
    w = data.weight
    by_state = (
        df.assign(w=w)
        .groupby("state")[["log_ratio", "w"]]
        .apply(lambda d: 100 * (np.exp(np.average(d.log_ratio, weights=d.w)) - 1))
    )
    return {
        "model": model,
        "areas_over_20pct_above": int((df.pct_vs_expected > 20).sum()),
        "areas_over_20pct_below": int((df.pct_vs_expected < -20).sum()),
        "share_of_population_within_10pct": float(
            w[df.pct_vs_expected.abs() <= 10].sum() / w.sum()
        ),
        "most_above_expected": top(by.tail(TOP_N)[::-1]),
        "most_below_expected": top(by.head(TOP_N)),
        "moran_log_ratio": spatial.global_moran(log_ratio, data.w),
        "lisa_clusters": spatial.cluster_counts(lisa),
        "pct_vs_expected_by_state": by_state.sort_values(ascending=False).to_dict(),
    }, df


def train(table_path: Path | None = None, reports_dir: Path | None = None) -> dict:
    from dap.health import figures

    set_seed()
    table = load_table(table_path)
    data = model_data(table)
    reports = Path(reports_dir or paths.reports_dir())

    auto, lisa = autocorrelation(data)
    compare, oof = compare_models(data)
    best = max(
        ("ridge", "spatial_lag", "lightgbm"), key=lambda m: compare[m]["spatial"]["r2"]["mean"]
    )
    resid, resid_df = residuals(data, oof[best], best, table.state_name)
    interp, shap = explain(data)
    sens = sensitivity(table)

    results = {
        "target": {
            "name": "Potentially preventable hospitalisations, age-standardised rate per 100,000",
            "year": "2023-24",
            "modelled_as": "log(rate); errors on the rate scale after exp(), population-weighted",
            "areas": len(data),
            "sa4_groups": int(data.meta.sa4_code.nunique()),
            "features": int(data.X.shape[1]),
        },
        "cv": {"folds": N_SPLITS, "repeats": N_REPEATS, "schemes": list(SCHEMES), "seed": SEED},
        "spatial_autocorrelation": auto,
        "models": compare,
        "best_model_spatial_cv": best,
        "interpretation": interp,
        "residuals": resid,
        "sensitivity": sens,
    }
    results = sig(results)
    results_path = reports / RESULTS_NAME
    write_json(results_path, results)
    if reports_dir is None:  # a run into the real reports folder also refreshes the README numbers
        readme.update(results)

    out = table.loc[data.X.index, ["sa3_name", "sa4_name", "state_name", "gcc_code", "geometry"]]
    out = out.join(resid_df.drop(columns=["sa3_name", "state"])).join(
        lisa.cluster.rename("pph_cluster")
    )
    resid_path = paths.processed_dir() / RESIDUALS_NAME
    gpd.GeoDataFrame(out, crs=table.crs).reset_index().to_file(
        resid_path, layer="sa3", driver="GPKG"
    )

    figs = figures.make_all(table, data, lisa, resid_df, results, shap, reports / "figures")
    return {"results": results_path, "residuals": resid_path, "figures": figs, "stats": results}
