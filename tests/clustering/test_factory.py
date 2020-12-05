import pytest

from directional_clustering.clustering import ClusteringFactory
from directional_clustering.clustering import ClusteringAlgorithm

# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def factory():
    """
    A factory that creates clustering algorithms.
    """
    return ClusteringFactory


@pytest.fixture
def supported_algorithms():
    """
    A list with the names of all the supported clustering algorithms.
    TODO: This should be automatically queried, not hand-written!
    """
    supported = ["variational kmeans", "cosine kmeans"]
    return supported


@pytest.fixture
def unsupported_algorithm():
    """
    The name of a weird, definitely unsupported clustering algorithm.
    """
    return "unsupported clustering"


@pytest.fixture
def new_invalid_algorithm():
    """
    The name of new clustering algorithm that is not API-compliant.
    """
    class InvalidClustering():
        pass

    return "new invalid clustering", InvalidClustering

# ==============================================================================
# Tests
# ==============================================================================

def test_create_supported_algorithms(supported_algorithms, factory):
    """
    Checks that all supported algorithms are created without raising errors.
    """
    for name in supported_algorithms:
        algorithm = factory.create(name)
        assert isinstance(algorithm, type(ClusteringAlgorithm))


def test_create_unsupported_algorithm_fails(unsupported_algorithm, factory):
    """
    Raises an error as the algorithm is not registered in the factory.
    """
    with pytest.raises(KeyError):
        factory.create(unsupported_algorithm)


def test_create_new_algorithm_fails(new_invalid_algorithm, factory):
    """
    Registration of an algorithm that is not a subclass of ClusteringAlgorithm fails.
    """
    name, algorithm = new_invalid_algorithm
    with pytest.raises(AssertionError):
        factory.register(name, algorithm)
