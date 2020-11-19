from compas.geometry import add_vectors
from compas.geometry import subtract_vectors
from compas.geometry import scale_vector


__all__ = ["smoothen_vector_field",
           "adjacent_vectors",
           "mean_vector",
           "smoothed_vector"]


def smoothen_vector_field(vector_field, adjacency, iters, damping=0.5):
    """
    Apply Laplacian smoothing to a vector field.

    Parameters
    ----------
    vector_field: `VectorField`
        A vector field.
    adjacency : `dict`
        A dictionary that maps a key to all the other keys neighboring it.
    iters : `int`
        The number of iterations to run this algorithm for.
    damping : `float`. Optional.
        A coefficient between 0.0 and 1.0 that controls the smoothing strenth.
        1.0 is maximum smoothing.
        Defaults to 0.5

    Notes
    -----
    Modifies vector field in place.
    """
    assert vector_field.size() == len(adjacency)

    for _ in range(iters):

        smoothed_vectors = {}

        # do one full round of laplacian smoothing
        for key in vector_field.keys():

            vector = vector_field.vector(key)
            neighbors = adjacency[key]

            if not neighbors:
                smoothed_vectors[key] = vector
                continue

            adj_vector = mean_vector(adjacent_vectors(vector_field, neighbors))

            smoothed_vectors[key] = smoothed_vector(vector, adj_vector, damping)

        # update vector field
        for key in vector_field.keys():
            vector_field.add_vector(key, smoothed_vectors[key])


def adjacent_vectors(vector_field, neighbors):
    """
    Query the vectors neighboring a vector field entry.
    """
    return [vector_field.vector(key) for key in neighbors]


def mean_vector(vectors):
    """
    Compute the mean of a sequence of vectors.
    """
    if not vectors:
        raise ValueError("Sequence of vectors is empty")

    m_vector = [0.0, 0.0, 0.0]

    for vector in vectors:
        m_vector = add_vectors(vector, m_vector)

    return scale_vector(m_vector, 1.0 / len(vectors))


def smoothed_vector(vector, s_vector, damping):
    """
    Apply Laplacian smoothing to a vector.
    """
    assert damping <= 1.0
    assert damping >= 0.0

    difference = subtract_vectors(s_vector, vector)
    s_vector = scale_vector(difference, 1.0 - damping)

    return add_vectors(vector, s_vector)


if __name__ == "__main__":
    pass
