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
    iters : `int`
        The iterations to run the algorithm for.
    tol : `float`
        The tolerance to declare convergence.
    """
    def __init__(self, mesh, vector_field):
        # sanity check
        assert mesh.number_of_faces() >= vector_field.size()

        # data structures
        # TODO: Make n_clusters an __init__ argument to unify seed()/cluster()?
        self.mesh = mesh
        self.vector_field = vector_field

        # distance function
        self.distance_func = None

        # initialization
        self.seeds = None

        # to be set after running cluster()
        self._clustered_field = None
        self._centers = None
        self._labels = None

        self._loss = None
        self._loss_history = None

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
    def loss_history(self):
        """
        The log of losses recorded while clustering a vector field.

        Returns
        -------
        loss : `list` of `float`
            The loss history.

        Notes
        -----
        The length of the list is the number of elapsed clustering iterations.
        """
        return self._loss_history

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

    @property
    def centers(self):
        """
        A mapping that maps cluster indices to cluster centroid vectors.

        Returns
        -------
        centers : `dict`
            A dictionary of the form `{cluster_index: centroid_vector}`.
        """
        return self._centers

    def cluster(self, n_clusters, iters, tol, early_stopping, *args, **kwargs):
        """
        Cluster a vector field.

        Parameters
        ----------
        n_clusters : `int`
            The number of clusters to generate.
        iters : `int`
            The number of iterations to run the k-means for.
        tol : `tol`
            The loss relative difference between iterations to declare early convergence.
        early_stopping : `bool`
            Flag to stop when tolerance threshold is met.
            Otherwise, the algorithm will exhaust all iterations.
        args : `list`, optional
            Additional arguments.
        kwargs : `dict`, optional
            Additional keyword arguments.

        Notes
        -----
        It sets `self._clustered_field`, `self_labels`, `self.centers`, and `self.loss`.
        Returns `None`.
        """
        # check sanity
        assert self.mesh.number_of_faces() >= n_clusters
        assert len(list(self.vector_field)) >= n_clusters

        # convert list of vectors to an array
        X = np.array(self.vector_field.to_sequence())

        # fetch initial seeds
        seeds = self.seeds
        if seeds is None:
            raise ValueError("No initial seeds! Have you created them?")

        # perform the clustering
        labels, centers, losses = self._cluster(X,
                                                self.seeds,
                                                self.distance_func,
                                                n_clusters,
                                                iters,
                                                tol,
                                                early_stopping,
                                                *args,
                                                **kwargs)

        # fetch assigned centroid to each entry in the vector field
        clusters = centers[labels]

        # create a new vector field
        clustered_field = VectorField()
        clustered_labels = {}

        # fill in vector field with clustering results
        for index, fkey in enumerate(self.vector_field.keys()):
            vector = clusters[index, :].tolist()
            clustered_field.add_vector(fkey, vector)
            clustered_labels[fkey] = labels[index]

        # store data as clustering object attributes
        self._clustered_field = clustered_field  # clustered vector field
        self._labels = clustered_labels  # face labels
        self._centers = {idx: center.tolist() for idx, center in enumerate(centers)}
        self._loss = losses[-1]
        self._loss_history = losses

    @staticmethod
    def _cluster(X, W, dist_func, n_clusters, iters, tol, early_stopping, *args, **kwargs):
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
        n_clusters : `int`
            The number of clusters to generate.
        iters : `float`
            The number of iterations to run the k-means for.
        tol : `tol`
            The loss relative difference between iterations to declare early convergence.
        early_stopping : `bool`
            Flag to stop when tolerance threshold is met.
            Otherwise, the algorithm will exhaust all iterations.
        args : `list`, optional
            Additional arguments.
        kwargs : `dict`, optional
            Additional keyword arguments.

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
        losses = []
        for i in range(iters):

            # assign labels
            loss, labels = centroids_associate(X, W, dist_func)
            losses.append(loss)

            # recalculate centroids
            W = centroids_estimate(X, n_clusters, labels)

            # check for exit
            if i < 2 or not early_stopping:
                continue

            # check if relative loss difference between two iterations is small
            eps = (losses[-2] - losses[-1]) / losses[-1]
            if fabs(eps) < tol:
                print(f"Convergence threshold met: {eps} < {tol}")
                print(f"Early stopping at {i}/{iters} iteration")
                break

        return labels, W, losses

    def seed(self, n_clusters, iters=100, tol=1e-6, early_stopping=False, *args, **kwargs):
        """
        Generate the initial cluster seeds using a farthest-point search.
        The first seed is picked at random, without replacement.

        Parameters
        ----------
        n_clusters : `int`
            The number of clusters to generate.
        iters : `float`, optional
            The number of iterations to run the k-means for.
            Defaults to `100`.
        tol : `tol`, optional
            The loss relative difference between iterations to declare early convergence.
            Defaults to `1e-6`.
        early_stopping : `bool`, optional
            Flag to stop when tolerance threshold is met.
            Otherwise, the algorithm will exhaust all iterations.
            Defaults to `False`.
        args : `list`, optional
            Additional arguments.
        kwargs : `dict`, optional
            Additional keyword arguments.

        Notes
        -----
        Returns `None`.
        This is a private method.
        """
        assert self.mesh.number_of_faces() >= n_clusters
        assert len(list(self.vector_field)) >= n_clusters

        # convert list of vectors to an array
        X = np.array(self.vector_field.to_sequence())

        # generate seeds and store them as self.seeds
        # TODO: early stopping is hard-coded as false for seed making
        seeds = self._seeds_generate(X,
                                     n_clusters,
                                     iters,
                                     tol,
                                     early_stopping,
                                     *args,
                                     **kwargs)
        self.seeds = seeds

    def _seeds_generate(self, X, n_clusters, iters, tol, early_stopping, *args, **kwargs):
        """
        Find the initial seeds using an iterative farthest-point search.
        The first seed is picked at random, without replacement.

        Parameters
        ----------
        X : `np.array`, (n, 3)
            An array with the vectors of a vector field.
        n_clusters : `int`
            The number of clusters to generate.
        iters : `float`
            The number of iterations to run the k-means for.
        tol : `tol`
            The loss relative difference between iterations to declare early convergence.
        early_stopping : `bool`
            Flag to stop when tolerance threshold is met.
            Otherwise, the algorithm will exhaust all iterations.
        args : `list`, optional
            Additional arguments.
        kwargs : `dict`, optional
            Additional keyword arguments.

        Returns
        -------
        seeds : `np.array`, (k, 3)
            An array with the initial cluster seeds.

        Notes
        -----
        This is a private method.
        """
        W = kmeans_initialize(X, 1, replace=False)

        for k in range(n_clusters - 1):
            labels, W = self._seeds_cluster(X,
                                            W,
                                            self.distance_func,
                                            k + 1,
                                            iters,
                                            tol,
                                            early_stopping,
                                            *args,
                                            **kwargs)

            values = W[labels]

            # TODO: Replace pairwise_distances with custom method
            # Check `sklearn.metrics.pairwise.pairwise_distances` for more info
            distances = pairwise_distances(X, values, metric=self.distance_name)
            distances = np.diagonal(distances).reshape(-1, 1)

            index = np.argmax(distances, axis=0)
            farthest = X[index, :]
            W = np.vstack([W, farthest])

        return W

    def _seeds_cluster(self, *args, **kwargs):
        """
        The clustering approach to create initial seeds.

        Parameters
        ----------
        args : `list`, optional
            Additional arguments.
        kwargs : `dict`, optional
            Keyword arguments.

        Returns
        -------
        labels : `np.array` (n, )
            The index of the center closest to every vector in the vector field.
        seeds : `np.array` (k, 3)
            The cluster seeds centers.

        Notes
        -----
        This is a private method.
        """
        labels, seeds, _ = self._cluster(*args, **kwargs)
        return labels, seeds


if __name__ == "__main__":
    pass
