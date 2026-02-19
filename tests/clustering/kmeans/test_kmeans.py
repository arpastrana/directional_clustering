import pytest

from directional_clustering.clustering import distance_cosine


# ==============================================================================
# Tests
# ==============================================================================

def test_loss_is_none(kmeans):
    """
    Checks that the loss is set to None before clustering.
    """
    assert kmeans.loss is None


def test_clustered_field_is_none(kmeans):
    """
    Checks that the clustered field is set to None prior to clustering.
    """
    assert kmeans.clustered_field is None


def test_labels_is_none(kmeans):
    """
    Asserts that the clustering labels are set to None before clustering.
    """
    assert kmeans.labels is None


@pytest.mark.parametrize("dist_func", [distance_cosine])
def test_static_clustering(dist_func, kmeans, vector_array, seed_array, n_clusters, iters, tol):
    """
    Checks that the clusters formed are correct based on a distance function.
    """
    result = kmeans._cluster(
        vector_array,
        seed_array,
        dist_func,
        loss_func=lambda x: x,
        n_clusters=n_clusters,
        iters=iters,
        tol=tol,
        early_stopping=False,
        is_seeding=False)

    # Expected length of result is 4
    assert len(result) == 4

    # unpack result
    labels, centers, _, _ = result

    # test labels
    assert labels.tolist() == [0, 0, 1], centers

    # assert centers
    assert centers.tolist() == [[0.0, 0.0, 1.5], [0.0, 2.0, 0.0]]
