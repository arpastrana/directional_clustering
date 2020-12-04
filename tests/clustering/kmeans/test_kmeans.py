import pytest

from directional_clustering.clustering import distance_cosine

# ==============================================================================
# Tests
# ==============================================================================

def test_loss_is_none(kmeans):
    """
    Checks that the loss of a KMeans object is None before running cluster()
    """
    assert kmeans.loss is None


def test_clustered_field_is_none(kmeans):
    """
    Checks that the clustered field in a KMeans object is None prior to clustering.
    """
    assert kmeans.clustered_field is None


def test_labels_is_none(kmeans):
    """
    Asserts that the clustering labels is set to  None before clustering.
    """
    assert kmeans.labels is None


@pytest.mark.parametrize("dist_func", [distance_cosine])
def test_static_clustering(dist_func, kmeans, vector_array, seed_array, n_clusters, iters, tol):
    """
    Check that the clusters formed are correct based on a distance function.
    """
    result = kmeans._cluster(vector_array, seed_array, dist_func, iters, tol)
    assert len(result) == 3

    # unpack result
    labels, centers, losses = result
    # test laels
    assert labels.tolist() == [0, 0, 1], centers
    # assert centers
    assert centers.tolist() == [[0.0, 0.0, 1.5], [0.0, 2.0, 0.0]]
