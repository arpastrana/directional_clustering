from directional_clustering.clustering.kmeans import KMeans
from directional_clustering.clustering.kmeans import DifferentiableKMeans

from directional_clustering.clustering.kmeans import distance_cosine
from directional_clustering.clustering.kmeans import distance_cosine_abs


__all__ = ["CosineKMeans",
           "CosineAbsoluteKMeans",
           "DifferentiableCosineKMeans",
           "DifferentiableCosineAbsoluteKMeans"]


class CosineKMeans(KMeans):
    """
    K-means clustering using cosine distance as the association metric.

    Parameters
    ----------
    mesh : `directional_clustering.mesh.MeshPlus`
        A reference mesh. Reserved.
    vector_field : `directional_clustering.fields.VectorField`
        The vector field to cluster.
    """
    def __init__(self, mesh, vector_field):
        # initialize parent class constructor
        super(CosineKMeans, self).__init__(mesh, vector_field)

        # set appropiate distance function
        self.distance_func = distance_cosine


class CosineAbsoluteKMeans(KMeans):
    """
    K-means clustering using absolute cosine distance as the association metric.

    Parameters
    ----------
    mesh : `directional_clustering.mesh.MeshPlus`
        A reference mesh. Reserved.
    vector_field : `directional_clustering.fields.VectorField`
        The vector field to cluster.
    """
    def __init__(self, mesh, vector_field):
        # initialize parent class constructor
        super(CosineAbsoluteKMeans, self).__init__(mesh, vector_field)

        # set appropiate distance function
        self.distance_func = distance_cosine_abs


class DifferentiableCosineKMeans(DifferentiableKMeans, CosineKMeans):
    """
    Differentiable k-means clustering using cosine distance as the kernel function.

    Parameters
    ----------
    mesh : `directional_clustering.mesh.MeshPlus`
        A reference mesh. Reserved.
    vector_field : `directional_clustering.fields.VectorField`
        The vector field to cluster.
    """
    def __init__(self, mesh, vector_field):
        # initialize parent class constructor
        super(DifferentiableCosineKMeans, self).__init__(mesh, vector_field)


class DifferentiableCosineAbsoluteKMeans(DifferentiableKMeans, CosineAbsoluteKMeans):
    """
    Differentiable k-means clustering using cosine absolute distance as the kernel function.

    Parameters
    ----------
    mesh : `directional_clustering.mesh.MeshPlus`
        A reference mesh. Reserved.
    vector_field : `directional_clustering.fields.VectorField`
        The vector field to cluster.
    """
    def __init__(self, mesh, vector_field):
        # initialize parent class constructor
        super(DifferentiableCosineAbsoluteKMeans, self).__init__(mesh, vector_field)


if __name__ == "__main__":
    pass
