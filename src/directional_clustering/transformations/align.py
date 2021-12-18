from compas.geometry import dot_vectors
from compas.geometry import scale_vector


__all__ = ["align_vector_field", "align_vectors"]


def align_vector_field(vector_field, reference_vector):
    """
    Aligns vectors to match the orientation of reference vector.

    Parameters
    ----------
    vector_field : `directional_clustering.fields.VectorField`
        A vector field.
    reference_vector : `list` of `float`
        The vector whose orientation is to be matched.

    Notes
    -----
    Comparison made with dot products.
    Modifies vector field in place.
    """
    for fkey, vector in vector_field.items():
        # if vectors don't point in the same direction, reverse it
        vector_field.add_vector(fkey, align_vectors(vector, reference_vector))


def align_vectors(vector, reference_vector):
    """
    Flip a vector so that in points in the same direction as the reference vector.

    Parameters
    ----------
    vector : `list` of `float`
        The vector to align.
    reference_vector : `list` of `float`
        The vector whose orientation is to be matched.

    Returns
    -------
    aligned_vector : `list` of `float`
        The aligned vector.
    """
    # if vectors don't point in the same direction, flip one of them
    if dot_vectors(vector, reference_vector) < 0.0:
        vector = scale_vector(vector, -1.0)
    return vector


if __name__ == "__main__":
    pass
