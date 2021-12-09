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
    """
    def __init__(self, mesh, vector_field, n_clusters, iters, tol):
        # check sanity
        assert mesh.number_of_faces() >= n_clusters
        assert len(list(vector_field)) >= n_clusters

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
        The total loss that k-means produced after clustering a vector field.

        Returns
        -------
        loss : `float`
            The loss.

        Notes
        -----
        This is computed as the mean squared distance of to the k-centers.
        """
        return self._loss

    @property
    def clustered_field(self):
        """
        The clustered vector field.

        Returns
        -------
        vector_field : `directional_clustering.fields.VectorField`
            The clustered vector field.

        Notes
        -----
        The vector field will contain k-unique vectors.
        This is a new object that leaves the original vector field unchanged.
        """
        return self._clustered_field

    @property
    def labels(self):
        """
        A mapping from a vector field's keys to the indices of the clusters centers.

        Returns
        -------
        labels : `dict`
            A dictionary of the form `{key: cluster_index}`.
        """
        return self._labels

    def _create_seeds(self, metric):
        """
        Find the initial seeds using an interative farthest-point strategy.
        The first seed is picked at random, without replacement.

        Parameters
        ----------
        metric : `str`
            The name of the distance metric to use.
            Available options are `"cosine"` and `"euclidean"`.
            Check `sklearn.metrics.pairwise.pairwise_distances` for more info.

        Notes
        -----
        This is a private method.
        It sets `self.seeds` and returns `None`.
        """
        X = np.array(self.vector_field.to_sequence())
        k = self.n_clusters
        replace = False

        W = kmeans_initialize(X, 1, replace)

        for _ in range(k - 1):
            labels, W, _ = self._cluster(X, W, self.distance_func, self.iters, self.tol, False)

            values = W[labels]

            # TODO: Replace pairwise_distances with custom method
            distances = pairwise_distances(X, values, metric=metric)
            distances = np.diagonal(distances).reshape(-1, 1)

            index = np.argmax(distances, axis=0)
            farthest = X[index, :]
            W = np.vstack([W, farthest])

        self.seeds = W

    def cluster(self):
        """
        Cluster a vector field.

        Notes
        -----
        It sets `self._clustered_field`, `self_labels`, `self.centers`, and `self.loss`.
        Returns `None`.
        """
        # convert list of vectors to an array
        X = np.array(self.vector_field.to_sequence())

        # perform the clustering
        r = self._cluster(X, self.seeds, self.distance_func, self.iters, self.tol, False)
        labels, centers, losses = r

        # fetch assigned centroid to each entry in the vector field
        clusters = centers[labels]

        # create a new vector field
        clustered_field = VectorField()
        clustered_labels = {}

        for index, fkey in enumerate(self.vector_field.keys()):
            vector = clusters[index, :].tolist()
            clustered_field.add_vector(fkey, vector)
            clustered_labels[fkey] = labels[index]

        # store data into attributes
        self._clustered_field = clustered_field  # clustered vector field
        self._labels = clustered_labels  # face labels
        self._centers = {idx: center for idx, center in enumerate(centers)}
        self._loss = losses[-1]

    @staticmethod
    def _cluster(X, W, dist_func, iters, tol, early_stopping=False):
        """
        Perform k-means clustering on a vector field.

        Parameters
        ----------
        X : `np.array`, (n, 3)
            An array with the vectors of a vector field.
        W : `np.array`, (k, 3)
            An array with the clusters' centers.
        dist_func : `function`
            A distance function to calculate the association metric.
        iters : `float`
            The number of iterations to run the k-means for.
        tol : `tol`
            The loss relative difference between iterations to declare early convergence.
        early_stopping : `bool`, optional.
            Flag to stop when tolerance threshold is met.
            Otherwise, the algorithm will exhaust all iterations.
            Defaults to `False`.

        Returns
        -------
        labels : `np.array` (n, )
            The index of the center closest to every vector in the vector field.
        W : `np.array` (k, 3)
            The cluster centers of a vector field.
        losses : `list` of `float`
            The losses generated at every iteration.

        Notes
        -----
        The nuts and bolts of kmeans clustering.
        This is a private method.
        """
        k, d = W.shape

        losses = []

        for i in range(iters):

            # assign labels
            loss, labels = centroids_associate(X, W, dist_func)
            losses.append(loss)

            # recalculate centroids
            W = centroids_estimate(X, k, labels)

            # check for exit
            if i < 2 or not early_stopping:
                continue

            # check if relative loss difference between two iterations is small
            if fabs((losses[-2] - losses[-1]) / losses[-1]) < tol:
                print("Early stopping at {}/{} iteration".format(i, iters))
                break

        return labels, W, losses


if __name__ == "__main__":
    pass
