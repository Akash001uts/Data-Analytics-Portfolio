"""One global seed for every random step, so runs are repeatable."""

import os
import random

import numpy as np

SEED = 20261006


def set_seed(seed: int = SEED) -> np.random.Generator:
    """Seed Python and NumPy, and return a NumPy generator for code that takes one explicitly."""
    random.seed(seed)
    np.random.seed(seed)  # some libraries still read NumPy's legacy global state
    os.environ["PYTHONHASHSEED"] = str(seed)
    return np.random.default_rng(seed)
