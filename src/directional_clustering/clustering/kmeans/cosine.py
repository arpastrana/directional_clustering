from directional_clustering.clustering.kmeans import KMeans

from directional_clustering.clustering.kmeans import distance_cosine


__all__ = ["CosineKMeans"]


class CosineKMeans(KMeans):
    """
    K-means clustering using cosine distance as association metric.
    """
    def __init__(self, mesh, vector_field, n_clusters, iters, tol):
        # initialize parent class constructor
        super(CosineKMeans, self).__init__(mesh, vector_field, n_clusters, iters, tol)

        # set appropiate distance function
        self.distance_func = distance_cosine

        # create seeds
        self._create_seeds("cosine")


if __name__ == "__main__":
    pass
