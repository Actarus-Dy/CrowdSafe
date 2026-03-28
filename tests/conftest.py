import numpy as np
import pytest


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def sample_pedestrians(rng):
    """100 pedestrians with random speeds and positions in a 50m x 50m venue."""
    n = 100
    dtype = np.dtype(
        [
            ("position_x", np.float64),
            ("position_y", np.float64),
            ("speed", np.float64),
            ("local_density", np.float64),
            ("v_max", np.float64),
            ("mass", np.float64),
        ]
    )
    pedestrians = np.zeros(n, dtype=dtype)
    pedestrians["position_x"] = rng.uniform(0, 50, n)
    pedestrians["position_y"] = rng.uniform(0, 50, n)
    pedestrians["speed"] = rng.uniform(0.5, 1.8, n)  # walking speeds m/s
    pedestrians["local_density"] = rng.uniform(0.5, 4.0, n)  # pers/m²
    pedestrians["v_max"] = 2.5  # running speed m/s
    pedestrians["mass"] = 0.0  # to be assigned
    return pedestrians
