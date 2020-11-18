#!/usr/bin/env python3

from compas.geometry import dot_vectors
from compas.geometry import scale_vector


__all__ = ["align_vector_field"]


def align_vector_field(vector_field, reference_vector):
    """
    Aligns vectors to match the orientation of reference vector.

    Parameters
    ----------
    vector_field: `VectorField`
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
        if dot_vectors(reference_vector, vector) < 0.0:
            vector_field.add_vector(fkey, scale_vector(vector, -1.0))
