from math import fabs

from numpy import array
from numpy import arange
from numpy import amax
from numpy import random
from numpy import reshape
from numpy import float64
from numpy import isnan
from numpy import square
from numpy import nonzero
from numpy import mean
from numpy import zeros
from numpy import reshape
from numpy import argmin
from numpy import argmax
from numpy import dot
from numpy import min
from numpy import abs
from numpy import log
from numpy import exp
from numpy import sum
from numpy import hstack
from numpy import vstack
from numpy import diagonal

from numpy.linalg import norm

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances

from time import time


__all__ = [
    "kmeans_clustering",
    "kmeans_fit",
    "init_kmeans_farthest",
    "_kmeans"
    ]


def kmeans_clustering(data, n_clusters, shape=None, normalize=False, random_state=0, n_jobs=-1):

    assert isinstance(data, dict)

    np_data = [x[1] for x in data.items()]
    np_data = array(np_data, dtype=float64)

    if normalize:
        np_data /= amax(np_data)

    if reshape:
        np_data = reshape(np_data, shape)

    print()
    print("fitting {} clusters".format(n_clusters))
    t0 = time()

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_jobs=n_jobs)
    km.fit(np_data)

    print("done in {} s, number of iterations: {}".format((time() - t0), km.n_iter_))
    return km.labels_, km.cluster_centers_


def rows_squared_norm(M, keepdims=False):
    """
    """
    return square(rows_norm(M, keepdims))


def rows_norm(M, keepdims=False):
    """
    """
    return norm(M, ord=2, axis=1, keepdims=keepdims)


def init_kmeans(X, k, replace=False):
    """
    Initialize k-means routine.

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
    n, d = X.shape
    bag = arange(0, n)
    indices = random.choice(bag, k, replace=replace)
    return X[indices]


def init_kmeans_farthest(X, k, dist="euclidean", epochs=20, eps=1e-3, replace=False):
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

    W = init_kmeans(X, 1, replace)

    for i in range(k-1):
        labels, W, _ = _kmeans(X, W, dist, epochs, eps, False, False)

        values = W[labels]

        distances = pairwise_distances(X, values, metric=dist)
        distances = diagonal(distances).reshape(-1, 1)
    
        index = argmax(distances, axis=0)
        farthest = X[index, :]
        W = vstack([W, farthest])

    return W


def associate_centroids_euclidean(X, W):
    """
    Input
    -----
    X : `np.array`, shape (n, d)
        2D value matrix where rows are examples.
    W : `np.array`, shape (k, d)
        2D value matrix where rows are k centroids.
    
    Returns
    -------
    loss : `np.array`
        The mean squared distance to all nearest centroids
    closest_k : `np.array`, shape(n,)
        The closest k-centroid for every example in X.
    """

    xn = rows_squared_norm(X)
    wn = rows_squared_norm(W)

    if isnan(wn).any():
        raise Exception("There's a NaN in wn: {}".format(wn))

    xn = reshape(xn, (-1, 1))
    wn = reshape(wn, (1, -1))

    sq_dist = xn + wn  # broadcasting should help. makes square matrix (n, k)

    sq_dist = sq_dist - 2 * dot(X, W.T)
    closest_k = argmin(sq_dist, axis=1)
    sq_dist = min(sq_dist, axis=1)

    return mean(sq_dist), closest_k


def associate_centroids_cosine(X, W):
    """
    Input
    -----
    X : `np.array`, shape (n, d)
        2D value matrix where rows are examples.
    W : `np.array`, shape (k, d)
        2D value matrix where rows are k centroids.
    
    Returns
    -------
    loss : `np.array`
        The mean squared distance to all nearest centroids
    closest_k : `np.array`, shape(n,)
        The closest k-centroid for every example in X.
    """

    xn = rows_norm(X, keepdims=True)
    wn = rows_norm(W, keepdims=True)

    if isnan(wn).any():
        raise Exception("There's a NaN in wn: {}".format(wn))

    X /= xn

    W /= wn
    W[nonzero(isnan(W))] = 0.0

    cos_similarity = dot(X, W.T)
    cos_distance = 1 - cos_similarity
    # distance = square(cosine_distance)
    distance = abs(cos_distance)

    closest_k = argmin(distance, axis=1)
    loss = mean(min(distance, axis=1))

    return loss, closest_k


def estimate_centroids(X, k, assoc):
    """
    Recalculates centroids.

    Inputs
    ------
    X : `np.array`, shape (n, d)
        Input data, where rows are examples and columns are features.
    k : `int`
        Number of k clusters to generate.
    assoc : `np.array`, shape (n, )
        Index of closest centroid for every example in X.

    Returns
    -------
    W  : `np.array`, shape (k, d)
        The new centroids recalculated based on previous associations.
    """
    n, d = X.shape
    W = zeros((k, d))

    for i in range(k):        
        associated = X[nonzero(assoc==i)]

        if 0 in associated.shape:  # to catch no example associated to i
            centroid = zeros((1, d))
        else:
            centroid = mean(associated, axis=0)  # mean over columns

        W[i] = centroid

    return W


def kmeans_fit(X, k, dist="euclidean", epochs=20, eps=1e-3, early_stopping=True, verbose=True):
    """
    """
    W = init_kmeans(X, k)  # initialize centroids

    return _kmeans(X, W, dist, epochs, eps, early_stopping, verbose)


def _kmeans(X, W, dist, epochs, eps, early_stopping, verbose):
    """
    """
    k, d = W.shape

    losses = []

    func_map = {
        "euclidean": associate_centroids_euclidean,
        "cosine": associate_centroids_cosine
    }

    for i in range(epochs):

        associator = func_map.get(dist)
        loss, assoc = associator(X, W)
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
    import numpy as np
    import math


    np.random.seed(4)

    for _ in range(10):
        a = np.random.rand(2, 2)
        b = rows_squared_norm(a)
        c = math.sqrt(a[0,0] ** 2 + a[0,1] ** 2) ** 2
        assert np.allclose(b[0], np.array([c])), "{} != {}".format(b[0], c)

        a = np.random.randn(4, 2)
        b = init_kmeans(a, k=2)
        assert all([b_ in a for b_ in b])

    X = np.linspace(1, 10, 10, dtype=np.float32).reshape(-1, 1)
    X = np.hstack([X, X])

    k = 3
    W = init_kmeans(X, k)
    assert W.shape[0] == k

    W = init_kmeans_farthest(X, k, dist="euclidean", replace=False, epochs=20, eps=1e-3)

    for _ in range(10):
        sq_dist, closest_k = associate_centroids_cosine(X, W)    
        W = estimate_centroids(X, k, closest_k)
    
    assoc, W, losses = kmeans_fit(X, k, epochs=20, eps=1e-3, early_stopping=False)

    for loss in losses:
         print(loss)
    
    print("Tests passed!")

    # ====

    import matplotlib.pyplot as plt
    from numpy import random as R
    import numpy as np

    def norm_rows(M):
        return M / np.sqrt(np.sum(M * M, axis=1, keepdims=True))


    def plot_sample(X, C, bb=[], ms=10, colors='bgrcmk'):
        if X.shape[1] > 2: # if there is more than 2 columns, project
            _, v = np.linalg.eig(X.T @ X)
            v = np.array(v, dtype=np.float32)
            # v[:,0:2] is the two most significant eigenvectors
            # X @ v[:,0:2] projects X onto these two eigenvectors
            Xplot = X @ v[:,0:2]
        else:  # otherwise, just plot the data as it is
            Xplot = X
        from itertools import cycle
        k = int(C.max()) + 1
        fig = plt.figure('Sample (Projected)', figsize=(8,8))
        if bb != []:
            plt.xlim(bb[0]), plt.ylim(bb[1])
        cycol = cycle(colors)
        for i in range(k):
            ind = C == i
            col = next(cycol)
            plt.scatter(Xplot[ind,0], Xplot[ind,1], s=ms, c=col)


    def generate_mixture_params(k, d, iso=True, sep=1):
        Pi = np.exp(R.randn(k) / np.sqrt(k))
        Pi = Pi / sum(Pi)
        Mu = sep * norm_rows(R.randn(k, d))
        Si  = [] 
        for i in range(k):
            A = R.randn(d, 3 * d) 
            A = A @ A.T
            A = np.linalg.det(A) ** (-1/k) * A # To prevent spikiness
            if iso:
                A = np.diag(np.diag(A))
            Si.append(A)
        Si = np.array(Si)
        return (Pi, Mu, Si)
    

    def sample_from_mog(mog, n):
        Pi, Mu, Si = mog
        k, d = Mu.shape
        s = R.multinomial(n, Pi)
        X = np.zeros((n, d))
        C = np.zeros(n)
        po = 0
        for i in range(k):
            pn = po + s[i]
            X[po:pn,:] = R.multivariate_normal(Mu[i], Si[i], s[i])
            C[po:pn] = i
            po = pn
        return C, X


    n, d, k = 10000, 20, 5
    mog = generate_mixture_params(k, d, iso=False, sep=np.sqrt(k/d))
    C, X = sample_from_mog(mog, n)
    plot_sample(X, C)
    plt.show()

    assoc, W1, loss = kmeans_fit(X, 1, verbose=False)
    print('[{:-2}/{:-2}] {:7.0}'.format(1, 1, loss[-1]))

    bl = loss[-1] * np.ones(3 * k)
    nz = np.zeros(3 * k, dtype=int)

    for test_k in np.arange(2, 2 * k + 1):
        
        for e in range(100):
            assoc, W, loss = kmeans_fit(X, test_k, verbose=False)
        
            if loss[-1] < bl[test_k]:
                bl[test_k] = loss[-1]
                nz[test_k] = (rows_squared_norm(W) != 0).sum().item()
        
        print('[{:-2}/{:-2}] {:-7.4}'.format(nz[test_k], test_k, bl[test_k].round(4)))
