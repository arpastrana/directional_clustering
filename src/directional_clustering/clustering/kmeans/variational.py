from variational_clustering.clustering import furthest_init
from variational_clustering.clustering import make_faces
from variational_clustering.clustering import k_means

from directional_clustering.clustering.kmeans import KMeans

from directional_clustering.fields import VectorField


__all__ = ["VariationalKMeans"]


class VariationalKMeans(KMeans):
    """
    The variational shape approximation method for vector clustering.

    Parameters
    ----------
    mesh : `directional_clustering.mesh.MeshPlus`
        A reference mesh.
    vector_field : `directional_clustering.fields.VectorField`
        The vector field to cluster.
    n_clusters : `int`
        The number of clusters to generate.
    iters : `int`
        The iterations to run the algorithm for.
    tol : `float`
        The tolerance to declare convergence.

    Notes
    -----
    This method normalizes all vectors before doing clustering.

    References
    ----------
    [1] Cohen-Steiner, D., Alliez, P., Desbrun, M. (2004). Variational Shape Approximation.
    RR-5371, INRIA. 2004, pp.29. inria-00070632
    """
    def __init__(self, mesh, vector_field, n_clusters, iters, tol):
        # parent class constructor
        args = mesh, vector_field, n_clusters, iters, tol
        super(VariationalKMeans, self).__init__(*args)

        # internal flag to control cluster splitting heuristic
        self.merge_split = True

        # to be set after initialization
        self._initial_clusters = None
        self._faces = None

    def cluster(self, n_clusters, iters, tol, early_stopping, *args, **kwargs):
        """
        Cluster a vector field.

        Parameters
        ----------
        n_clusters : `int`
            The number of clusters to generate.
        iters : `int`
            The number of iterations to run the k-means for.
        args : `list`, optional
            Additional arguments.
        kwargs : `dict`, optional
            Additional keyword arguments.

        Notes
        -----
        It sets `self._clustered_field`, `self_labels`, `self.centers`, and `self.loss`.
        Returns `None`.
        """
        # create seeds
        self._create_seeds(n_clusters)

        # do clustering
        cluster_log = k_means(self._initial_clusters,
                              self._faces,
                              iters,
                              self.merge_split)

        # last chunk in the cluster log
        final_clusters = cluster_log.pop()

        # create a new vector field
        clustered_field = VectorField()
        clustered_labels = {}
        centers = {}

        # fill arrays with results
        # TODO: Refactor this block!
        loss = 0
        for i, cluster in final_clusters.items():
            centroid = cluster.proxy
            centers[i] = centroid

            loss += cluster.distortion

            for fkey in cluster.faces_keys:
                clustered_field.add_vector(fkey, centroid)
                clustered_labels[fkey] = cluster.id

        # assign arrays as attributes
        self._clustered_field = clustered_field
        self._labels = clustered_labels
        self._centers = centers
        self._loss = loss

    def _create_seeds(self, n_clusters, *args, **kwargs):
        """
        Find the initial seeds for clustering using a farthest-point strategy.

        Parameters
        ----------
        n_clusters : `int`
            The number of clusters to generate.
        args : `list`, optional
            Additional arguments.
        kwargs : `dict`, optional
            Additional keyword arguments.

        Notes
        -----
        This is a private method.
        It internally sets `self._faces` and `self._initial_clusters`.
        Returns `None`.
        """
        vectors = {key: vector for key, vector in self.vector_field.items()}
        self._faces = make_faces(self.mesh, vectors)
        self._initial_clusters = furthest_init(n_clusters, self._faces).pop()
