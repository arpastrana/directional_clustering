import pytest

import numpy as np

from directional_clustering.clustering import cosine_distance


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def random_array():
    """
    A 2x2 array with random float values.
    """
    return np.random.rand(2, 2)

# ==============================================================================
# Tests
# ==============================================================================

@pytest.mark.parametrize("vector,result", [([1.0, 0.0], 0.0),
                                           ([-1.0, 0.0], 2.0),
                                           ([0.0, 1.0], 1.0)])
def test_cosine_distance(vector, result):
    """
    Regular case of distance of one vector to itself.
    """
    assert cosine_distance([1.0, 0.0], vector) == result


def test_cosine_distance_fails_shape(random_array):
    """
    Fails because the shapes of the arrays do not match.
    """
    m, n = random_array.shape
    with pytest.raises(ValueError):
        other_array = np.ones((m, n + 1))
        cosine_distance(random_array, other_array)
