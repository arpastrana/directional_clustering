import numpy as np

from math import fabs

from sklearn.metrics.pairwise import pairwise_distances

from directional_clustering.clusters import associate_centroids_cosine
from directional_clustering.clusters import estimate_centroids
from directional_clustering.clusters import init_kmeans

from directional_clustering.clustering import ClusteringAlgorithm


__all__ = [
    "CosineKMeans"
    ]


class CosineKMeans(ClusteringAlgorithm):
    """
    K-means clustering using cosine distance as association metric.
    """
    def __init__(self, mesh, vector_field, n_clusters, n_init, max_iter=100, tol=0.001):
        self.mesh = mesh  # unused, needs refactoring
        self.vector_field = vector_field

        self.n_clusters = n_clusters

        self.seeds_iter = n_init
        self.max_iter = max_iter
        self.tol = tol

        # to be set after initialization
        self.seeds = None

        # to be set after running cluster()
        self.clusters = None
        self.cluster_centers = None
        self.labels = None
        self.loss = None
        self.n_iter = None

        # create seeds
        self.init_seeds()

    def init_seeds(self):
        """
        Initialize k-means with a farthest-point strategy.

        Input
        -----
        X : `np.array`, shape (n, d)
            2D value matrix where rows are examples.
        k : `int`
            Number of clusters to generate.
        replace : `bool`
            Flag to sample with or without replacement.
            Defaults to `False`.

        Returns
        -------
        W : `np.array`, shape (k, d)
            Matrix with cluster centroids, sampled from `X w/o replacement.
        """
        X = self.vector_field
        k = self.n_clusters
        epochs = self.seeds_iter
        eps = self.tol
        replace = False

        W = init_kmeans(X, 1, replace)

        for i in range(k-1):
            labels, W, _ = self._cluster(X, W, epochs, eps, False, False)

            values = W[labels]

            distances = pairwise_distances(X, values, metric="cosine")
            distances = np.diagonal(distances).reshape(-1, 1)
        
            index = np.argmax(distances, axis=0)
            farthest = X[index, :]
            W = np.vstack([W, farthest])

        self.seeds = W
    
    def cluster(self):
        """
        Main clustering method

        Input
        -----

        Returns
        -------
        """
        r = self._cluster(self.vector_field, self.seeds, self.max_iter, self.tol)

        labels, centers, losses = r
        self.clusters = centers[labels]  # clustered vector field
        self.labels = labels  # face labels
        self.cluster_centers = centers  # clusters centroids
        self.loss = losses[-1]

    def _cluster(self, X, W, epochs, eps, early_stopping=True, verbose=True):
        """
        Internal clustering method

        Input
        -----

        Returns
        -------
        """
        k, d = W.shape

        losses = []

        for i in range(epochs):

            loss, assoc = associate_centroids_cosine(X, W)
            losses.append(loss)
            
            W = estimate_centroids(X, k, assoc)

            if i < 2 or not early_stopping:
                continue

            if fabs((losses[-2] - losses[-1]) / losses[-1]) < eps:
                if verbose:
                    print("Early stopping at {}/{} iteration".format(i, epochs))
                break

        return assoc, W, losses


if __name__ == "__main__":    
    pass
