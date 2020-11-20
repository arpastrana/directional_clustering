import numpy as np


__all__ = ["distance_cosine"]


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
    The cosine distance can be expressed 1 - AB.
    """
    return 1.0 - np.dot(A, np.transpose(B))


if __name__ == "__main__":
    pass
