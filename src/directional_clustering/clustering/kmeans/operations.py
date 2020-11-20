from math import fabs

from numpy import arange
from numpy import random
from numpy import isnan
from numpy import square
from numpy import nonzero
from numpy import mean
from numpy import zeros
from numpy import argmax
from numpy import min
from numpy import sum
from numpy import dot
from numpy import argmin
from numpy import transpose

from numpy.linalg import norm

from sklearn.metrics.pairwise import pairwise_distances



__all__ = ["kmeans_initialize",
           "centroids_associate",
           "centroids_estimate",
           "cosine_distance"]


def rows_squared_norm(M):
    """
    Calculate the squared norm of the rows of a 2d matrix.

    Parameters
    ----------
    M : `np.array`, (n, d)
        A 2D array.

    Returns
    -------
    squared_norm : `np.array`, (n, 1)
        A 2d array with the squared norm of the rows.
    """
    return square(rows_norm(M))


def rows_norm(M):
    """
    Calculate the norm of the rows of a 2d matrix.

    Parameters
    ----------
    M : `np.array`, (n, d)
        A 2D array.

    Returns
    -------
    norm : `np.array`, (n, 1)
        A 2d array with the norm of the rows.
    """
    return norm(M, ord=2, axis=1, keepdims=True)


def kmeans_initialize(X, k, replace=False):
    """
    Picks k values at random from a 2d matrix.

    Parameters
    ----------
    X : `np.array`, (n, d)
        2D array where rows are examples.
    k : `int`
        Number of clusters to generate.
    replace : `bool`, (optional)
        Flag to sample with or without replacement.
        Defaults to `False`.

    Returns
    -------
    W : `np.array`, (k, d)
        Matrix with cluster centroids, sampled from `X` w/o replacement.
    """
    n, d = X.shape

    assert k <= n, "Number of clusters is larger than number of values to cluster!"

    bag = arange(0, n)
    indices = random.choice(bag, k, replace=replace)
    return X[indices]


def centroids_estimate(X, k, assoc):
    """
    Calculate new centroids based on an association vector.

    Parameters
    ----------
    X : `np.array`, (n, d)
        Input data, where rows are examples and columns are features.
    k : `int`
        Number of k clusters to generate.
    assoc : `np.array`, (n, )
        Index of closest centroid for every example in X.

    Returns
    -------
    W  : `np.array`, (k, d)
        The new centroids recalculated based on previous associations.
    """
    n, d = X.shape
    W = zeros((k, d))

    for i in range(k):
        associated = X[nonzero(assoc == i)]

        if 0 in associated.shape:  # to catch no example associated to i
            centroid = zeros((1, d))
        else:
            centroid = mean(associated, axis=0)  # mean over columns

        W[i] = centroid

    return W


def centroids_associate(X, W, d_func):
    """
    Parameters
    ----------
    X : `np.array`, (n, d)
        2D array where rows are examples.
    W : `np.array`, (k, d)
        2D array where rows are the k centroids.
    d_func : `function`
        A function to calculate distances with.

    Returns
    -------
    loss : `np.array`
        The mean squared distance to all nearest centroids
    closest_k : `np.array`, (n,)
        The closest k-centroid for every example in X.
    """

    # compute the norms of the rows
    xn = rows_norm(X)
    wn = rows_norm(W)

    # check for nan's
    if isnan(wn).any():
        raise Exception("There's a NaN in wn: {}".format(wn))

    # normalize rows
    X = X / xn
    W = W / wn

    # set nan's to zero for numerical stability
    W[nonzero(isnan(W))] = 0.0

    # compute distance
    distance = d_func(X, W)

    # find closest indices
    closest_k = argmin(distance, axis=1)

    # calculate mean squared error
    loss = mean(min(square(distance), axis=1))

    return loss, closest_k


def cosine_distance(A, B):
    """
    Computes the cosine distance between two arrays.

    Parameters
    ----------
    A : `np.array` (n, d)
        The first array.
    B : `np.array` (k, d)
        The second array.

    Returns
    -------
    distance : `np.array` (n, k)
        The distance of each entry in `A` (rows) to every entry in `B` (columns).

    Notes
    -----
    The cosine distance can be expressed 1 - AB.
    """
    return 1.0 - dot(A, transpose(B))


if __name__ == "__main__":

    X = np.linspace(1, 10, 10, dtype=np.float32).reshape(-1, 1)
    X = np.hstack([X, X])

    k = 3
    W = kmeans_initialize(X, k)
    assert W.shape[0] == k

    for _ in range(10):
        sq_dist, closest_k = centroids_associate(X, W)
        W = centroids_estimate(X, k, closest_k)
