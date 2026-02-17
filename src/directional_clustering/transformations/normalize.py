from compas.geometry import normalize_vector


__all__ = ["normalize_vector_field"]


def normalize_vector_field(vector_field):
    """
    Makes vectors to be all of unit length.

    Parameters
    ----------
    vector_field : `directional_clustering.fields.VectorField`
        A vector field.

    Notes
    -----
    Modifies vector field in place.
    """
    for fkey, vector in vector_field.items():
        vector_field.add_vector(fkey, normalize_vector(vector))


if __name__ == "__main__":
    pass
