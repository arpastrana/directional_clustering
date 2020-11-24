from directional_clustering.clustering.kmeans import KMeans

from variational_clustering.clustering import furthest_init
from variational_clustering.clustering import make_faces
from variational_clustering.clustering import k_means

from directional_clustering.fields import VectorField


__all__ = ["VariationalKMeans"]


class VariationalKMeans(KMeans):
    """
    A wrapper of the variational approximation algorithm for vector clustering.
    """
    def __init__(self, mesh, vector_field, n_clusters, iters, tol):
        # parent class constructor
        args = mesh, vector_field, n_clusters, iters, tol
        super(VariationalKMeans, self).__init__(*args)

        # internal flag
        self.merge_split = True

        # to be set after initialization
        self._initial_clusters = None
        self._faces = None

        # create seeds
        self._create_seeds()

    def cluster(self):
        """
        Simplify a vector field.
        """
        # do clustering
        cluster_log = k_means(self._initial_clusters,
                              self._faces,
                              self.iters,
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
        self._loss = loss  # implement better

    def _create_seeds(self):
        """
        Kickstart the algorithm.
        """
        self._faces = make_faces(self.mesh, self.vector_field)
        self._initial_clusters = furthest_init(self.n_clusters, self._faces).pop()
