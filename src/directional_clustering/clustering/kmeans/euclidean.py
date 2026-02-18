from directional_clustering.clustering.kmeans import KMeans

from directional_clustering.clustering.kmeans import distance_euclidean


__all__ = ["EuclideanKMeans"]


class EuclideanKMeans(KMeans):
    """
    K-means clustering using Euclidean distance as the association metric.

    Parameters
    ----------
    mesh : `directional_clustering.mesh.MeshPlus`
        A reference mesh. Reserved.
    vector_field : `directional_clustering.fields.VectorField`
        The vector field to cluster.
    """

    def __init__(self, mesh, vector_field):
        # initialize parent class constructor
        super(EuclideanKMeans, self).__init__(mesh, vector_field)

        # set appropiate distance function
        self.distance_func = distance_euclidean

if __name__ == "__main__":
    pass
