from math import fabs

import numpy as np

from directional_clustering.clustering.kmeans import CosineKMeans

from directional_clustering.clustering.kmeans import centroids_associate
from directional_clustering.clustering.kmeans import centroids_estimate


__all__ = ["DifferentiableCosineKMeans"]


class DifferentiableCosineKMeans(CosineKMeans):
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
        # parent classes sets distance function and creates initial seeds
        super(DifferentiableCosineKMeans, self).__init__(mesh, vector_field)

        # set attention parameters
        self.attention = None

    def _cluster(self, X, W, dist_func, n_clusters, iters, tol, early_stopping, tau, stabilize=True, *args, **kwargs):
        """
        Perform differentiable k-means clustering on a vector field.

        Parameters
        ----------
        X : `np.array`, (n, 3)
            An array with the vectors of a vector field.
        W : `np.array`, (k, 3)
            An array with the initial cluster centers.
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
        tau : `float`
            An coefficient that controls the softness of the attention mechanism
        stabilize : `bool`, optional
            A flag to numerically stabilize the attention softmax operation.
            Defaults to `True`.
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
        print("Starting differentiable k-means clustering...")
        k, r = W.shape
        n, d = X.shape
        assert d == r
        centroids = W

        losses = []

        for i in range(iters):

            # assign labels (centroids associate)
            # set nan's to zero for numerical stability
            centroids[np.nonzero(np.isnan(W))] = 0.0

            # compute distance matrix (n, k)
            distances = self.distance_func(X, centroids) * -1.0
            assert distances.shape == (n, k)

            # calculate attention matrix (n, k)
            z = distances * tau  # apply attention temperature
            if stabilize:
                z = z - np.amax(z)  # softmax stabilization
            z = np.exp(z)
            y = np.sum(z, axis=1, keepdims=True)
            assert y.shape == (n, 1)
            attention = z / y
            assert attention.shape == (n, k)
            assert np.isclose(np.sum(attention), n), print(np.sum(attention))

            # compute centroid matrix (k, d)
            centroids = np.transpose(attention) @ X
            centroids = centroids / np.transpose(np.sum(attention, axis=0, keepdims=True))
            assert centroids.shape == (k, d)

            # TODO: is MSE the right metric to calculate loss?
            # calculate root mean squared distance (error)
            closest = np.amin(np.square(distances), axis=1, keepdims=True)
            loss = np.sqrt(np.mean(closest))
            losses.append(loss)

            # check for exit
            if i < 2 or not early_stopping:
                continue

            # check if relative loss difference between two iterations is small
            eps = (losses[-2] - losses[-1]) / losses[-1]
            if fabs(eps) < tol:
                print(f"Convergence threshold met: {eps} < {tol}")
                print(f"Early stopping at {i}/{iters} iteration")
                break

        # create new temporary vector field to check clusters
        X_hat = attention @ centroids
        assert X_hat.shape == (n, d)

        # clip cluster assignment to closest centroid
        c_distances = self.distance_func(X_hat, centroids)
        labels = np.argmin(c_distances, axis=1)

        # sets attention matrix as clusterer attribute before exiting
        attention_dict = {}
        for index, fkey in enumerate(self.vector_field.keys()):
            attention_dict[fkey] = attention[index, :].tolist()

        self.attention = attention_dict

        print("Differentiable clustering ended!")
        return labels, centroids, losses

    def _cluster_seeds(self, *args, **kwargs):
        """
        The clustering approach to create initial seeds.

        Parameters
        ----------
        args : `list`
            A list of input parameters.
        kwargs : `dict`
            Named attributes

        Returns
        -------
        labels : `np.array` (n, )
            The index of the center closest to every vector in the vector field.
        seeds : `np.array` (k, 3)
            The cluster seeds centers.

        Notes
        -----
        Invokes parent classes initialization approach.
        This is a private method.
        """
        # Parent class cluster method to dettach it from attention mechanism
        labels, seeds, _ = super(DifferentiableCosineKMeans, self)._cluster(*args, **kwargs)
        return labels, seeds


if __name__ == "__main__":

    from compas.datastructures import Mesh

    from directional_clustering.clustering import CosineKMeans
    from directional_clustering.fields import VectorField

    vector_field = [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 2.0, 0.0]]
    vector_field = VectorField.from_sequence(vector_field)

    # creating dummy mesh to comply with API
    mesh = Mesh()
    for i in range(10):
        mesh.add_vertex()
    for i in range(vector_field.size()):
        mesh.add_face([0, 1, 2])

    n_clusters = 2  # number of clusters, TODO: sensitive if k=vector_field.size()?
    iters = 100  # max iterations
    tol = 1e-6  # convergence threshold
    early_stopping = False
    tao = 100  # attention temperature - small values (like 1) make centroids to equalize

    print("----")
    print("Clustering with Differentiable Cosine KMeans")

    dclusterer = DifferentiableCosineKMeans(mesh, vector_field)
    dclusterer.cluster(n_clusters, iters, tol, early_stopping, tao)
    dclustered = dclusterer.clustered_field


    print(f"Loss: {dclusterer.loss}, Labels: {dclusterer.labels}")
    print(f"Cluster centers {dclusterer.centers}")
    print(f"Clustered vector field {list(dclustered.vectors())}")

    print("----")
    print("Clustering with Cosine KMeans")

    clusterer = CosineKMeans(mesh, vector_field)
    clusterer.cluster(n_clusters, iters, tol, early_stopping)
    clustered = clusterer.clustered_field

    print(f"Loss: {clusterer.loss}, Labels: {clusterer.labels}")
    print(f"Cluster centers {clusterer.centers}")
    print(f"Clustered vector field {list(clustered.vectors())}")

    # assert dclusterer.labels == clusterer.labels

    assert clustered.vector(0) == [0.0, 0.0, 1.5]
    assert clustered.vector(1) == [0.0, 0.0, 1.5]
    assert clustered.vector(2) == [0.0, 2.0, 0.0]

    print("Saul Goodman!")
