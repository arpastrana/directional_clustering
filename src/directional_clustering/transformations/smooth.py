from compas.geometry import add_vectors
from compas.geometry import subtract_vectors
from compas.geometry import scale_vector

from directional_clustering.transformations import align_vectors


__all__ = ["smoothen_vector_field",
           "adjacent_vectors",
           "mean_vector",
           "smoothed_vector"]


def smoothen_vector_field(vector_field, adjacency, iters, damping=0.5, align=True):
    """
    Apply Laplacian smoothing to a vector field.

    Parameters
    ----------
    vector_field : `directional_clustering.clustering.VectorField`
        A vector field.
    adjacency : `dict`
        A dictionary that maps a key to all the other keys neighboring it.
    iters : `int`
        The number of iterations to run this algorithm for.
    damping : `float`, optional.
        A coefficient between 0.0 and 1.0 that controls the smoothing strength.
        1.0 is maximum smoothing.
        Defaults to 0.5.
    align : `bool`, optional.
        A flag to align adjacent vectors to any given vector before aggregation.
        The alignment check is carried via the dot product.
        Defaults to `True`.

    Notes
    -----
    Modifies the vector field in place.
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

            adj_vectors = adjacent_vectors(vector_field, neighbors)

            if align:
                adj_vectors = [align_vectors(v, vector) for v in adj_vectors]

            avg_vector = mean_vector(adj_vectors)

            smoothed_vectors[key] = smoothed_vector(vector, avg_vector, damping)

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
