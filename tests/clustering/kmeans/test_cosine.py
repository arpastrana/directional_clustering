import pytest

from directional_clustering.fields import VectorField

from directional_clustering.clustering import distance_cosine


# ==============================================================================
# Tests
# ==============================================================================

def test_distance_function(cosine_kmeans):
    """
    Checks that cosine kmeans uses distance_cosine as the distance calculator.
    """
    assert isinstance(cosine_kmeans.distance_func, type(distance_cosine))


def test_loss(cosine_kmeans, n_clusters, iters, tol):
    """
    Verifies that the clustering loss is zero for the test vector field.
    """
    cosine_kmeans.seed(n_clusters)
    cosine_kmeans.cluster(n_clusters, iters, tol, early_stopping=False)
    assert cosine_kmeans.loss == 0.0


def test_clustered_field_type(cosine_kmeans, n_clusters, iters, tol):
    """
    Asserts that the type of the clustered field is correct.
    """
    cosine_kmeans.seed(n_clusters)
    cosine_kmeans.cluster(n_clusters, iters, tol, early_stopping=False)

    assert isinstance(cosine_kmeans.clustered_field, VectorField)


def test_clustered_field_entries(cosine_kmeans, n_clusters, iters, tol):
    """
    Tests that the clustered field contains the right vectors.
    """
    cosine_kmeans.seed(n_clusters)
    cosine_kmeans.cluster(n_clusters, iters, tol, early_stopping=False)
    clustered = cosine_kmeans.clustered_field

    assert isinstance(clustered, VectorField)

    assert clustered.vector(0) == [0.0, 0.0, 1.5]
    assert clustered.vector(1) == [0.0, 0.0, 1.5]
    assert clustered.vector(2) == [0.0, 2.0, 0.0]


def test_labels(cosine_kmeans, n_clusters, iters, tol):
    """
    Checks that the vectors are associated with the right centroid.
    """
    cosine_kmeans.seed(n_clusters)
    cosine_kmeans.cluster(n_clusters, iters, tol, early_stopping=False)
    labels = cosine_kmeans.labels

    assert labels[0] == 0
    assert labels[1] == 0
    assert labels[2] == 1
