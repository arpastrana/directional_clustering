from math import fabs

import numpy as np

from sklearn.metrics.pairwise import pairwise_distances

from directional_clustering.clustering.kmeans import centroids_associate
from directional_clustering.clustering.kmeans import centroids_estimate
from directional_clustering.clustering.kmeans import kmeans_initialize

from directional_clustering.clustering import ClusteringAlgorithm

from directional_clustering.fields import VectorField


__all__ = ["KMeans"]


class KMeans(ClusteringAlgorithm):
    """
    Generic K-means clustering algorithm.
    """
    def __init__(self, mesh, vector_field, n_clusters, iters, tol):
        # data structures
        self.mesh = mesh
        self.vector_field = vector_field

        # tresholds
        self.iters = iters
        self.tol = tol

        # number of clusters to make
        self.n_clusters = n_clusters

        # distance function
        self.distance_func = None

        # initialization
        self.seeds = None

        # to be set after running cluster()
        self._clustered_field = None
        self._centers = None
        self._labels = None

        self._loss = None

    @property
    def loss(self):
        """
        """
        return self._loss

    @property
    def clustered_field(self):
        """
        """
        return self._clustered_field

    @property
    def labels(self):
        """
        """
        return self._labels

    def _create_seeds(self, metric):
        """
        """
        X = np.array(self.vector_field.to_sequence())
        k = self.n_clusters
        replace = False

        W = kmeans_initialize(X, 1, replace)

        for _ in range(k-1):
            labels, W, _ = self._cluster(X, W, self.iters, self.tol, False)

            values = W[labels]

            distances = pairwise_distances(X, values, metric=metric)
            distances = np.diagonal(distances).reshape(-1, 1)

            index = np.argmax(distances, axis=0)
            farthest = X[index, :]
            W = np.vstack([W, farthest])

        self.seeds = W

    def cluster(self, *args, **kwargs):
        """
        Main clustering method

        Parameters
        ----------

        Returns
        -------
        """
        X = np.array(self.vector_field.to_sequence())  # TODO
        r = self._cluster(X, self.seeds, self.iters, self.tol, False)

        labels, centers, losses = r
        clusters = centers[labels]

        # create a new vector field
        clustered_field = VectorField()
        clustered_labels = {}

        for fkey, index in self.vector_field.key_index().items():
            vector = clusters[index, :].tolist()
            clustered_field.add_vector(fkey, vector)
            clustered_labels[fkey] = labels[index]

        # store data into attributes
        self._clustered_field = clustered_field  # clustered vector field
        self._labels = clustered_labels  # face labels
        self._centers = {idx: center for idx, center in enumerate(centers)}
        self._loss = losses[-1]

    def _cluster(self, X, W, iters, tol, early_stopping=True):
        """
        Internal clustering method

        Parameters
        ----------

        Returns
        -------
        """
        k, d = W.shape

        losses = []

        for i in range(iters):

            # assign labels
            loss, assoc = centroids_associate(X, W, self.distance_func)
            losses.append(loss)

            # recalculate centroids
            W = centroids_estimate(X, k, assoc)

            # check for exit
            if i < 2 or not early_stopping:
                continue

            # check if relative loss difference between two iterations is too small
            if fabs((losses[-2] - losses[-1]) / losses[-1]) < tol:
                print("Early stopping at {}/{} iteration".format(i, iters))
                break

        return assoc, W, losses


if __name__ == "__main__":    
    pass
