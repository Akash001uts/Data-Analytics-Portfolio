"""Spatial weights and spatial autocorrelation: Queen contiguity, global Moran's I and LISA.

Weights are built on the areas that are actually modelled (the SA3s with a published target), so
dropping suppressed areas can't leave a hidden gap. Any area with no neighbour is joined to its
nearest neighbour rather than dropped.
"""

import contextlib
import io
import logging
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal.weights import KNN, Queen, W, attach_islands, w_subset

from dap.common.seeds import SEED

log = logging.getLogger(__name__)

PROJECTED_CRS = "EPSG:3577"  # GDA94 Australian Albers, metres; only used for nearest neighbours
PERMUTATIONS = 999
ALPHA = 0.05
QUADRANTS = {1: "High-high", 2: "Low-high", 3: "Low-low", 4: "High-low"}
NOT_SIGNIFICANT = "Not significant"


def queen_weights(geo: gpd.GeoDataFrame) -> W:
    """Row-standardised Queen contiguity weights, ids = the frame's index, islands attached."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # libpysal warns about islands and components
        w = Queen.from_dataframe(geo, ids=list(geo.index), use_index=False)
        if w.islands:
            log.info("attaching %d islands to their nearest neighbour", len(w.islands))
            pts = geo.to_crs(PROJECTED_CRS).geometry.centroid
            knn = KNN.from_dataframe(gpd.GeoDataFrame(geometry=pts), k=1, ids=list(geo.index))
            w = attach_islands(w, knn)
    w.transform = "r"
    return w


def subset_weights(w: W, ids: list) -> W:
    """Weights restricted to some areas and row-standardised again (used inside a training fold)."""
    # Cutting the map down leaves some areas with no neighbours; libpysal prints each one.
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        sub = w_subset(w, ids)
        sub.transform = "r"
    return sub


def describe_weights(w: W) -> dict:
    return {
        "areas": int(w.n),
        "mean_neighbours": float(w.mean_neighbors),
        "islands": len(w.islands),
        "components": int(w.n_components),
    }


def global_moran(y: pd.Series, w: W, permutations: int = PERMUTATIONS) -> dict:
    """Global Moran's I with a permutation p-value, with `y` in the order of `w.id_order`."""
    from esda import Moran

    y = _aligned(y, w)
    np.random.seed(SEED)  # esda's global Moran draws its permutations from NumPy's global state
    m = Moran(y.to_numpy(), w, permutations=permutations)
    return {
        "I": float(m.I),
        "expected_I": float(m.EI),
        "z_sim": float(m.z_sim),
        "p_sim": float(m.p_sim),
        "permutations": permutations,
    }


def local_moran(y: pd.Series, w: W, permutations: int = PERMUTATIONS) -> pd.DataFrame:
    """LISA: each area's local Moran's I, its quadrant and a cluster label at the 5% level.

    The p-values are not adjusted for multiple comparisons, so the clusters are a guide to where
    to look, not a list of confirmed hot spots.
    """
    from esda import Moran_Local

    y = _aligned(y, w)
    lm = Moran_Local(y.to_numpy(), w, permutations=permutations, seed=SEED)
    quad = pd.Series(lm.q, index=y.index).map(QUADRANTS)
    sig = pd.Series(lm.p_sim < ALPHA, index=y.index)
    return pd.DataFrame(
        {
            "local_i": lm.Is,
            "p_sim": lm.p_sim,
            "quadrant": quad,
            "cluster": quad.where(sig, NOT_SIGNIFICANT),
        },
        index=y.index,
    )


def cluster_counts(lisa: pd.DataFrame) -> dict:
    counts = lisa.cluster.value_counts()
    return {k: int(counts.get(k, 0)) for k in [*QUADRANTS.values(), NOT_SIGNIFICANT]}


def _aligned(y: pd.Series, w: W) -> pd.Series:
    if list(y.index) != list(w.id_order):
        y = y.reindex(w.id_order)
    if y.isna().any():
        raise ValueError("values are missing for some areas in the weights")
    return y
