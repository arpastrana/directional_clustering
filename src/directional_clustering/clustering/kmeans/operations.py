import numpy as np


__all__ = ["kmeans_initialize",
           "centroids_associate",
           "centroids_estimate"]


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
    return np.square(rows_norm(M))


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
    return np.linalg.norm(M, ord=2, axis=1, keepdims=True)


def kmeans_initialize(X, k, replace=False):
    """
    Picks k values at random from a 2d matrix with or without replacement.

    Parameters
    ----------
    X : `np.array`, (n, d)
        2D array where rows are examples.
    k : `int`
        Number of clusters to generate.
    replace : `bool`, optional
        Flag to sample with or without replacement.
        Defaults to `False`.

    Returns
    -------
    W : `np.array`, (k, d)
        Matrix with cluster centroids, sampled from `X` w/o replacement.
    """
    n, d = X.shape

    assert k <= n, "Number of clusters is larger than number of values to cluster!"

    bag = np.arange(0, n)
    indices = np.random.choice(bag, k, replace=replace)
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
    W = np.zeros((k, d))

    for i in range(k):
        associated = X[np.nonzero(assoc == i)]

        if 0 in associated.shape:  # to catch no example associated to i
            centroid = np.zeros((1, d))
        else:
            centroid = np.mean(associated, axis=0)  # mean over columns

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
    loss : `float`
        The mean squared distance to all nearest centroids
    closest_k : `np.array`, (n,)
        The closest k-centroid for every example in X.
    """
    # set nan's to zero for numerical stability
    W[np.nonzero(np.isnan(W))] = 0.0

    # compute distance
    distances = d_func(X, W)

    # find closest indices
    closest_k = np.argmin(distances, axis=1)

    # calculate mean absolute distance (error)
    closest = np.amin(np.abs(distances), axis=1, keepdims=True)
    loss = np.mean(closest)

    return loss, closest_k


if __name__ == "__main__":
    pass
