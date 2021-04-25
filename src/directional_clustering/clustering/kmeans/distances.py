import numpy as np


__all__ = ["distance_cosine", "distance_cosine_abs", "distance_euclidean"]


def distance_cosine(A, B):
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
    Cosine distance is bounded between 0 and 2, where 2 means the most distant.
    The cosine distance can be expressed 1 - cosine similarity.
    The cosine similarity is given by AB / (||A||||B||)
    """
    cos_sim = np.dot(A, np.transpose(B)) / (np.linalg.norm(A) * np.linalg.norm(B))

    return 1.0 - cos_sim


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
    The absolute cosine distance is bounded between 0 and 1. 1 means the most distant.
    The absolute cosine distance can be expressed 1 - abs(cosine similarity).
    The cosine similarity is given by AB / (||A||||B||)
    """
    cos_sim = np.dot(A, np.transpose(B)) / (np.linalg.norm(A) * np.linalg.norm(B))

    return 1.0 - np.abs(cos_sim)


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
