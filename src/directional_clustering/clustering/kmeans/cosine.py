from directional_clustering.clustering.kmeans import KMeans

from directional_clustering.clustering.kmeans import distance_cosine


__all__ = ["CosineKMeans"]


class CosineKMeans(KMeans):
    """
    K-means clustering using cosine distance as the association metric.

    Parameters
    ----------
    mesh : `directional_clustering.mesh.MeshPlus`
        A reference mesh. Reserved.
    vector_field : `directional_clustering.fields.VectorField`
        The vector field to cluster.
    n_clusters : `int`
        The number of clusters to generate.
    iters : `int`
        The iterations to run the algorithm for.
    tol : `float`
        The tolerance to declare convergence.
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
