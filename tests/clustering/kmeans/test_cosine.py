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


def test_loss(cosine_kmeans):
    """
    Verifies that the clustering loss is zero for the test vector field.
    """
    cosine_kmeans.cluster()
    assert cosine_kmeans.loss == 0.0


def test_clustered_field_type(cosine_kmeans):
    """
    Asserts that the type of the clustered field is correct.
    """
    cosine_kmeans.cluster()

    assert isinstance(cosine_kmeans.clustered_field, VectorField)


def test_clustered_field_entries(cosine_kmeans):
    """
    Tests that the clustered field contains the right vectors.
    """
    cosine_kmeans.cluster()
    clustered = cosine_kmeans.clustered_field

    assert isinstance(clustered, VectorField)

    assert clustered.vector(0) == [0.0, 0.0, 1.5]
    assert clustered.vector(1) == [0.0, 0.0, 1.5]
    assert clustered.vector(2) == [0.0, 2.0, 0.0]


def test_labels(cosine_kmeans):
    """
    Checks that the vectors are associated with the right centroid.
    """
    cosine_kmeans.cluster()
    labels = cosine_kmeans.labels

    assert labels[0] == 0
    assert labels[1] == 0
    assert labels[2] == 1
