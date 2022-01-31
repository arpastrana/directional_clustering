# i am too smart to manually compute gradients
from autograd import grad

# numpy on steroids
import autograd.numpy as np

# optimization is never a bad idea
from scipy.optimize import minimize


__all__ = ["DifferentiableKMeans"]


class DifferentiableKMeans():
    """
    Differentiable k-means clustering using a custom kernel function.

    Parameters
    ----------
    mesh : `directional_clustering.mesh.MeshPlus`
        A reference mesh. Reserved.
    vector_field : `directional_clustering.fields.VectorField`
        The vector field to cluster.
    """
    def __init__(self, mesh, vector_field):
        # initialize parent class constructor
        # TODO: Make n_clusters an __init__ argument to unify seed()/cluster()?
        super(DifferentiableKMeans, self).__init__(mesh, vector_field)
        # set attention parameters
        self.attention = None

    def _cluster(self, X, W, dist_func, loss_func, n_clusters, iters, tol, early_stopping, tau, stabilize=True, optimize=True, optimizer="BFGS", *args, **kwargs):
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
        loss_func : `function`
            A distance function to compute the value of the loss function.
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

        # soft dimensions-matching test
        assert d == r

        # create recorder to store values from clustering method
        recorder = {"attention": None, "centroids": None, "losses": [], "losses_field": []}

        # convert tau to array
        # TODO: does it require type check before reconverting to array?
        tau = np.array(tau)

        if optimize:
            # solve clustering problem via optimization
            res = self._cluster_optimize(X, W, iters, tol, early_stopping, tau, stabilize, optimizer)
            xopt = res.x
            tau = xopt
            print(f"Optimization solution: {res.x}.\nOutput message: {res.message}")

        # run clustering again with optimal parameters to overcome autograd issues with autograd values?
        loss = self._cluster_diff(X, W, iters, tol, early_stopping, tau, stabilize, recorder)
        print(f"Loss final run: {loss}.")

        # unpack recorder
        attention = recorder["attention"]
        centroids = recorder["centroids"]
        closest_k = recorder["closest_k"]
        losses = recorder["losses"]
        losses_field = recorder["losses_field"]

        # sets attention matrix as clusterer attribute before exiting
        attention_dict = {}
        for index, fkey in enumerate(self.vector_field.keys()):
            attention_dict[fkey] = attention[index, :].tolist()
        self.attention = attention_dict

        print("Differentiable clustering ended!")
        return closest_k, centroids, losses, losses_field

    def _cluster_optimize(self, X, centroids, iters, tol, early_stopping, tau, stabilize, optimizer):
        """
        Let the hunger games begin.
        """
        x0 = tau

        def func(x0, X, centroids, iters, tol, early_stopping, stabilize):
            """
            Helper function to re-organize ordering of inputs.
            The first argument is the optimization variable. Must be an array.
            """
            tau = x0
            last_loss = self._cluster_diff(X, centroids, iters, tol, early_stopping, tau, stabilize, recorder=None, verbose=False)

            return last_loss

        # gradient w.r.t. first argument to be compatible with scipy's function signature.
        grad_func = grad(func, argnum=0)

        # minimize func
        args = (X, centroids, iters, tol, early_stopping, stabilize)
        res = minimize(func, x0,
                       args=args,
                       jac=grad_func,
                       method=optimizer,  # BFGS, SLSQP
                       options={'disp': True, 'maxiter': 100, 'gtol': 1e-6})
        return res

    def _cluster_diff(self, X, centroids, iters, tol, early_stopping, tau, stabilize, recorder=None, verbose=True):
        """
        """
        losses = []
        losses_field = []

        for i in range(iters):

            # compute distance matrix (n, k)
            distances = self.distance_func(X, centroids)

            # find vectors' closest centroid
            closest_k = np.argmin(distances, axis=1)  # shape (n, )

            # compute loss of vector to its closest centroid (like in not-diff KMeans)
            # get distance of vector to its closest centroid
            closest_dist = distances[np.arange(len(closest_k)), closest_k]
            # TODO: Does slicing make gradients become zero? Related to copy/not copy?
            # closest_dist = distances
            loss = self.loss_func(closest_dist)

            # calculate attention matrix (n, k)
            attention = self._attention_matrix(distances, tau, stabilize)

            # compute centroid matrix (k, d)
            centroids = self._centroid_matrix(X, attention)

            # calculate loss from unclustered to clustered field
            X_hat = attention @ centroids
            loss_field = self.loss_func(self.distance_func(X, X_hat, row_wise=True))

            # store losses
            losses.append(loss)
            losses_field.append(loss_field)

            # check for exit
            if i < 2 or not early_stopping:
                continue

            # check if relative loss difference between two iterations is small
            eps = np.abs((losses[-2] - losses[-1]) / losses[-1])
            if eps < tol:
                if verbose:
                    if not isinstance(eps, float):
                        eps = eps._value
                    print(f"Convergence threshold met: {eps} < {tol}")
                    print(f"Early stopping at {i}/{iters} iteration")
                break

        if recorder:
            recorder["losses"] = losses
            recorder["losses_field"] = losses_field
            recorder["centroids"] = centroids
            recorder["closest_k"] = closest_k
            recorder["attention"] = attention

        return loss

    @staticmethod
    def _cluster_loss(distances, closest_k, loss_func):
        """
        """
        # get distance of vector to its closest centroid
        closest_dist = distances[np.arange(len(closest_k)), closest_k]
        # TODO: slicing creates issues with differentiability: gradients are zero

        # compute loss of vector to its closest centroid (like in not-diff KMeans)
        return loss_func(closest_dist)

    @staticmethod
    def _attention_matrix(distances, tau, stabilize):
        """
        Compute the attention matrix.
        """
        # calculate attention matrix (n, k)
        n, k = distances.shape
        # TODO: in the paper, why are distances multiplied times -1?
        # related to gaussian kernel, the exponential wants a negative input
        z = distances * tau * -1.0  # apply attention temperature
        if stabilize:
            z = z - np.amax(z)  # softmax stabilization
        z = np.exp(z)
        # y = np.sum(z, axis=1, keepdims=True)
        y = np.reshape(np.sum(z, axis=1), (n, 1))
        assert y.shape == (n, 1)
        attention = z / y
        assert attention.shape == (n, k)
        # assert np.isclose(np.sum(attention), n), print(np.sum(attention))
        return attention

    @staticmethod
    def _centroid_matrix(X, attention):
        """
        Estimate centroids given the input vectors and an attention matrix.
        """
        centroids = np.transpose(attention) @ X
        # centroids_sum = np.sum(attention, axis=0, keepdims=True)
        # print(np.sum(attention, axis=0).shape)
        # print(np.sum(attention, axis=0, keepdims=True).shape)
        centroids_sum = np.reshape(np.sum(attention, axis=0), (-1, 1))
        centroids = centroids / centroids_sum
        # centroids = centroids / np.transpose(np.sum(attention, axis=0, keepdims=True))
        # assert centroids.shape == (k, d)
        return centroids

    def _seeds_cluster(self, *args, **kwargs):
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
        labels, seeds, _, _ = super(DifferentiableKMeans, self)._cluster(*args, **kwargs)
        return labels, seeds


if __name__ == "__main__":

    import autograd.numpy as np
    from autograd import grad

    from compas.datastructures import Mesh

    from directional_clustering.clustering import DifferentiableCosineKMeans
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
    early_stopping = True
    tau = np.array([100.0]) # attention temperature - small values (like 1) make centroids to equalize
    stabilize = True
    # TODO: gradients seem to work only when argunum=0, and distance slicing is disabled
    argnum = 0

    print("----")
    print("Clustering with Differentiable Cosine KMeans")

    clusterer = DifferentiableCosineKMeans(mesh, vector_field)
    clusterer.seed(n_clusters, early_stopping=True)
    clusterer.cluster(n_clusters, iters, tol, early_stopping, tau)
    clustered = clusterer.clustered_field

    print(f"Loss: {clusterer.loss}, Labels: {clusterer.labels}")
    print(f"Cluster centers {clusterer.centers}")
    print(f"Clustered vector field {list(clustered.vectors())}")

    # assert np.allclose(clustered.vector(0), [0.0, 0.0, 1.5])
    # assert np.allclose(clustered.vector(1), [0.0, 0.0, 1.5])
    # assert np.allclose(clustered.vector(2), [0.0, 2.0, 0.0])

    recorder = {"attention": None,
                "centroids": None,
                "losses": [],
                "losses_field": []}

    grad_func = grad(clusterer._cluster_diff, argnum=argnum)

    X = np.array(vector_field.to_sequence())
    seeds = clusterer.seeds

    grad_cluster = grad_func(X, seeds, iters, tol, early_stopping, tau, stabilize, recorder)

    print(grad_cluster)
    print("Saul Goodman!")
