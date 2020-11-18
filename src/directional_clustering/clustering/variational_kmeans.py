"""
A wrapper around the variational clustering algorithm.
"""
from directional_clustering.clustering import ClusteringAlgorithm

from variational_clustering.clustering import furthest_init
from variational_clustering.clustering import make_faces
from variational_clustering.clustering import k_means

from directional_clustering.fields import VectorField


__all__ = ["VariationalKMeans"]


class VariationalKMeans(ClusteringAlgorithm):
    """
    A wrapper around the variational shape approximation algorithm.
    """
    def __init__(self, mesh, vector_field, n_clusters, n_init, max_iter=100, tol=0.001, merge_split=True):
        self.mesh = mesh
        self.vector_field = vector_field

        self.n_clusters = n_clusters

        self.seeds_iter = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.merge_split = merge_split

        # to be set after initialization
        self.seeds = None
        self._initial_clusters = None
        self._faces = None

        # to be set after running cluster()
        self.clusters = None
        self.cluster_centers = None
        self.labels = None
        self.loss = None
        self.n_iter = None

        # create seeds
        self.init_seeds()

    def cluster(self):
        """
        Simplify a vector field.
        """
        # do clustering
        cluster_log = k_means(self._initial_clusters,
                              self._faces,
                              self.max_iter,
                              self.merge_split)

        # last chunk in the cluster log
        final_clusters = cluster_log.pop()

        # create a new vector field
        clustered_field = VectorField()
        clustered_labels = {}
        centers = {}

        # fill arrays with results
        loss = 0
        for i, cluster in final_clusters.items():
            centroid = cluster.proxy
            centers[i] = centroid

            loss += cluster.distortion

            for fkey in cluster.faces_keys:
                clustered_field.add_vector(fkey, centroid)
                clustered_labels[fkey] = cluster.id

        # assign arrays as attributes
        self.clusters = clustered_field
        self.labels = clustered_labels
        self.cluster_centers = centers
        self.loss = loss  # implement better

    def init_seeds(self):
        """
        Kickstart the algorithm.
        """
        self._faces = make_faces(self.mesh, self.vector_field)
        self._initial_clusters = furthest_init(self.n_clusters, self._faces).pop()
