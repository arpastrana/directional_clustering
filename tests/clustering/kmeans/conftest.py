import pytest

import numpy as np


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def random_array():
    """
    A 2x2 array with random float values.
    """
    return np.random.rand(2, 2)


@pytest.fixture
def cosine_array():
    """
    A 3x2 array with float values.
    """
    return np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


@pytest.fixture
def cosine_centroids():
    """
    A 2x2 array of floats that represents the centroids of two 2D clusters.
    """
    return np.array([[1.0, 0.0], [1.0, 1.0]])
