# import numpy as np
import autograd.numpy as np


__all__ = ["distance_cosine",
           "distance_cosine_abs",
           "distance_euclidean"]


def cosine_similarity(A, B, row_wise):
    """
    Computes the cosine distance between two arrays.
    """
    def rows_norm(M):
        brn = np.linalg.norm(M, ord=2, axis=1)
        brn = np.reshape(brn, (M.shape[0], -1))
        return brn

    A = np.atleast_2d(A)
    B = np.atleast_2d(B)

    A = A / rows_norm(A)
    B = B / rows_norm(B)

    if row_wise:
        # both matrices must have equal number of rows
        assert A.shape[0] == B.shape[0]
        # return row-wise dot product
        return np.einsum('ij,ij->i', A, B)

    else:
        return np.dot(A, np.transpose(B))


def distance_cosine(A, B, row_wise=False):
    """
    Computes the cosine distance between two arrays.

    Parameters
    ----------
    A : `np.array` (n, d)
        The first array.
    B : `np.array` (k, d)
        The second array.
    row_wise : `bool`, optional.
        If `True`, it calculates pairwise distances between every row in A and every possible row in B.
        Otherwise, distances are calculated only between matching rows assuming `n=k`.
        Defaults to `False`.

    Returns
    -------
    distance : `np.array` (n, k)
        The distance of each entry in `A` (rows) to every entry in `B` (columns).

    Notes
    -----
    Cosine distance is bounded between 0 and 2, where 2 means the most distant.
    The cosine distance can be expressed 1 - cosine similarity.
    The cosine similarity is given by AB / (||A||||B||)
    """
    return 1.0 - cosine_similarity(A, B, row_wise)


def distance_cosine_abs(A, B):
    """
    Computes the absolute cosine distance between two arrays.
    Here the absolute value of the cosine similarity is taken.

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
    The absolute cosine distance is bounded between 0 and 1.
    1 means the most distant.
    The absolute cosine distance can be expressed 1 - abs(cosine similarity).
    The cosine similarity is given by AB / (||A||||B||)
    """
    return 1.0 - np.abs(cosine_similarity(A, B))


def distance_euclidean(A, B):
    """
    Computes the Euclidean distance between two arrays.

    Parameters
    ----------
    A : `np.array` (n, d)
        The first array.
    B : `np.array` (k, d)
        The second array.

    Returns
    -------
    distance : `np.array` (n, k)

    Notes
    -----
    The distance of every row in `A` to every row in `B`.
    Every entry at (row, col) = (r, c) represents the Euclidean distance
    between A[r, :] and B[c, :].
    """
    # TODO: Test this function.
    R = np.sum(A * A, axis=1, keepdims=True)
    C = np.sum(B * B, axis=1, keepdims=True)
    C = np.transpose(C)
    G = A @ np.transpose(B)
    D = R + C - 2 * G

    return np.sqrt(D)


if __name__ == "__main__":
    assert distance_euclidean(np.reshape(np.array([-1.0, 0.0, 0.0]), (1, -1)),
                              np.reshape(np.array([3.0, 0.0, 0.0]), (1, -1))) == 4.0

    assert distance_cosine(np.reshape(np.array([-1.0, 0.0, 0.0]), (1, -1)),
                           np.reshape(np.array([3.0, 0.0, 0.0]), (1, -1))) == 2.0

    assert distance_cosine_abs(np.reshape(np.array([-1.0, 0.0, 0.0]), (1, -1)),
                               np.reshape(np.array([3.0, 0.0, 0.0]), (1, -1))) == 0.0
