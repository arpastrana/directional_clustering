import pytest

import numpy as np

from directional_clustering.fields import VectorField

from directional_clustering.clustering import distance_cosine


# ==============================================================================
# Tests
# ==============================================================================

def test_loss(variational_kmeans, n_clusters, iters, tol, seed_array):
    """
    Verifies that the clustering loss is zero for a test vector field.
    """
    variational_kmeans.seeds = seed_array
    variational_kmeans.cluster(n_clusters, iters, tol, early_stopping=False)
    assert variational_kmeans.loss == 0.0


def test_clustered_field_type(variational_kmeans, n_clusters, iters, tol, seed_array):
    """
    Asserts that the type of the clustered field is correct.
    """
    variational_kmeans.seeds = seed_array
    variational_kmeans.cluster(n_clusters, iters, tol, early_stopping=False)

    assert isinstance(variational_kmeans.clustered_field, VectorField)


def test_clustered_field_entries(variational_kmeans, n_clusters, iters, tol, seed_array):
    """
    Tests that the clustered field contains the right vectors.
    """
    variational_kmeans.seeds = seed_array
    variational_kmeans.cluster(n_clusters, iters, tol, early_stopping=False)
    clustered = variational_kmeans.clustered_field

    assert isinstance(clustered, VectorField)

    sqrt_2 = (2**0.5) * 0.5
    assert np.allclose(np.array(clustered.vector(0)), np.array([0.0, 0.0, 1.0]))
    assert np.allclose(np.array(clustered.vector(1)), np.array([0.0, sqrt_2, sqrt_2]))
    assert np.allclose(np.array(clustered.vector(2)), np.array([0.0, sqrt_2, sqrt_2]))


def test_labels(variational_kmeans, n_clusters, iters, tol, seed_array):
    """
    Checks that the vectors are associated with the right centroid.
    """
    variational_kmeans.seeds = seed_array
    variational_kmeans.cluster(n_clusters, iters, tol, early_stopping=False)
    labels = variational_kmeans.labels

    assert labels[0] == 1
    assert labels[1] == 0
    assert labels[2] == 0


if __name__ == "__main__":
    pass
