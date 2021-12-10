import pytest

from directional_clustering.fields import VectorField

from directional_clustering.clustering import distance_cosine


# ==============================================================================
# Tests
# ==============================================================================

def test_distance_function(diff_cosine_kmeans):
    """
    Checks that diff cosine kmeans uses distance_cosine as the distance calculator.
    """
    assert isinstance(diff_cosine_kmeans.distance_func, type(distance_cosine))


# def test_loss(diff_cosine_kmeans):
#     """
#     Verifies that the clustering loss is zero for the test vector field.
#     """
#     diff_cosine_kmeans.cluster()
#     assert diff_cosine_kmeans.loss == 0.0


def test_clustered_field_type(diff_cosine_kmeans):
    """
    Asserts that the type of the clustered field is correct.
    """
    diff_cosine_kmeans.cluster()

    assert isinstance(diff_cosine_kmeans.clustered_field, VectorField)


# def test_clustered_field_entries(diff_cosine_kmeans):
#     """
#     Tests that the clustered field contains the right vectors.
#     """
#     diff_cosine_kmeans.cluster()
#     clustered = diff_cosine_kmeans.clustered_field

#     assert isinstance(clustered, VectorField)

#     assert clustered.vector(0) == [0.0, 0.0, 1.5]
#     assert clustered.vector(1) == [0.0, 0.0, 1.5]
#     assert clustered.vector(2) == [0.0, 2.0, 0.0]


# def test_labels(diff_cosine_kmeans):
#     """
#     Checks that the vectors are associated with the right centroid.
#     """
#     diff_cosine_kmeans.cluster()
#     labels = diff_cosine_kmeans.labels

#     assert labels[0] == 0
#     assert labels[1] == 0
#     assert labels[2] == 1
