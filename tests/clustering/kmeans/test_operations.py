from math import sqrt

import pytest

import numpy as np

from directional_clustering.clustering import kmeans_initialize
from directional_clustering.clustering import centroids_associate
from directional_clustering.clustering import centroids_estimate
from directional_clustering.clustering import cosine_distance

from directional_clustering.clustering.kmeans.operations import rows_norm
from directional_clustering.clustering.kmeans.operations import rows_squared_norm


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
    """
    return np.array([[1.0, 0.0], [1.0, 1.0]])


# ==============================================================================
# Tests
# ==============================================================================


def test_rows_norm(random_array):
    """
    Check squared norm is calculated along the right axis.
    """
    random_norm = rows_norm(random_array).reshape(-1,)

    manual_norm = [random_array[0, 0]**2 + random_array[0, 1]**2,
                   random_array[1, 0]**2 + random_array[1, 1]**2]
    manual_norm = [sqrt(i) for i in manual_norm]

    assert np.allclose(random_norm, manual_norm), (random_norm, manual_norm)


def test_rows_norm_shape(random_array):
    """
    Check squared norm is calculated along the right axis.
    """
    random_norm = rows_norm(random_array)
    assert random_norm.shape == (random_array.shape[0], 1)


def test_rows_squared_norm(random_array):
    """
    Check squared norm is calculated along the right axis.
    """
    random_norm = rows_squared_norm(random_array).reshape(-1,)

    manual_norm = [random_array[0, 0] ** 2 + random_array[0, 1]**2,
                   random_array[1, 0] ** 2 + random_array[1, 1]**2]

    assert np.allclose(random_norm, manual_norm), (random_norm, manual_norm)


@pytest.mark.parametrize("size", list(range(1, 2)))
def test_kmeans_initialize_size(random_array, size):
    """
    Check that number of examples produced is correct.
    """
    seeds = kmeans_initialize(random_array, size)
    assert seeds.shape[0] == size


def test_kmeans_initialize_belongs(random_array):
    """
    Check that seeds exist in the array.
    """
    seed = kmeans_initialize(random_array, 1)
    assert seed in random_array


def test_kmeans_initialize_bad_k(random_array):
    """
    Tries to produce more clusters than entries in the random array.
    """
    n, d = random_array.shape
    with pytest.raises(AssertionError):
        kmeans_initialize(random_array, n+1)


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


def test_centroids_associate_cosine(cosine_array, cosine_centroids):
    """
    Assign labels to three vectors based on two redundant centroids.
    """
    _, closest = centroids_associate(cosine_array,
                                     cosine_centroids,
                                     cosine_distance)

    assert np.allclose(closest, np.array([0, 1, 1])), closest


def test_centroids_estimate_cosine(cosine_array):
    """
    Re-estimate two new centroids based on redundant association.
    """
    assoc = np.array([0, 1, 1])
    new_centroids = centroids_estimate(cosine_array, 2, assoc)
    assert np.allclose(new_centroids, np.array([[1.0, 0.0], [0.5, 1.0]]))
