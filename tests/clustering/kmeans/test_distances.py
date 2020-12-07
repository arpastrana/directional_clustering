import pytest

import numpy as np

from directional_clustering.clustering import distance_cosine


# ==============================================================================
# Tests
# ==============================================================================

@pytest.mark.parametrize("vector,result", [([1.0, 0.0], 0.0),
                                           ([-1.0, 0.0], 2.0),
                                           ([0.0, 1.0], 1.0)])
def test_distance_cosine(vector, result):
    """
    Regular case of distance of one vector to itself.
    """
    assert distance_cosine([1.0, 0.0], vector) == result


def test_distance_cosine_fails_shape(random_array):
    """
    Fails because the shapes of the arrays do not match.
    """
    m, n = random_array.shape
    with pytest.raises(ValueError):
        other_array = np.ones((m, n + 1))
        distance_cosine(random_array, other_array)
