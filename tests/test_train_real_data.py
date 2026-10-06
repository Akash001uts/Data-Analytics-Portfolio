"""Checks on the real modelling table. Skipped unless `dap health build` has been run."""

import numpy as np
import pytest

from dap.common import paths
from dap.health.build import TABLE_NAME
from dap.health.evaluate import splits
from dap.health.features import feature_names

_table = paths.processed_dir() / TABLE_NAME

pytestmark = [
    pytest.mark.skipif(not _table.exists(), reason="processed table not built"),
    pytest.mark.real_data,
]


@pytest.fixture(scope="module")
def data():
    from dap.health.train import load_table, model_data

    return model_data(load_table())


def test_modelling_rows(data):
    assert len(data) == 327
    assert data.meta.sa4_code.nunique() == 88
    assert list(data.X.columns) == feature_names(("A",))
    assert data.weight.gt(0).all() and np.isfinite(data.y).all()


def test_weights_cover_every_area(data):
    assert data.w.n == len(data) and not data.w.islands


@pytest.mark.parametrize("repeat", range(5))
def test_no_sa4_in_both_train_and_test(data, repeat):
    sa4 = data.meta.sa4_code.to_numpy()
    for train, test in splits(data, "spatial", repeat):
        assert not set(sa4[train]) & set(sa4[test])
