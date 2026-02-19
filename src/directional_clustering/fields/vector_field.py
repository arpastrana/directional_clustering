from directional_clustering.fields import Field


__all__ = ["VectorField"]


class VectorField(Field):
    """
    A field with a fixed dimensionality of 3.
    # TODO: Any field should be a VectorField and this should be renamed to 3DVectorField?
    """
    def __init__(self):
        """
        The constructor.
        """
        super(VectorField, self).__init__(dimensionality=3)

    # --------------------------------------------------------------------------
    # Vector operations
    # --------------------------------------------------------------------------

    def add_vector(self, key, vector):
        """
        Adds a vector entry to the field.

        Parameters
        ----------
        key : `int`
            The key to store the vector with.
        vector : `list` of `float`
            A vector in 3d space.
        """
        self[key] = vector

    def remove_vector(self, key):
        """
        Deletes a vector from the field.

        Parameters
        ----------
        key : `int`
            The key of the vector to remove.
        """
        del self[key]

    def vector(self, key):
        """
        Queries a vector from a field.

        Parameters
        ----------
        key : `int`
            The key of the vector to retrieve.

        Returns
        -------
        vector : `list` of `float`
            A vector.
        """
        return self[key]

    def keys(self):
        """
        Iterates over they access keys of the field.

        Yields
        ------
        key : `int`
            The next access key in the field.
        """
        for key, _ in self:
            yield key

    def vectors(self):
        """
        Iterates over the vectors of the field.

        Yields
        ------
        vector : `list` of `float`
            The next vector in the field.
        """
        for _, vector in self:
            yield vector

    def items(self):
        """
        Iterates over the keys and the vectors of the field.

        Yields
        ------
        key : `int`
            The next access key in the field.
        vector : `list` of `float`
            The next vector in the field.
        """
        for key, vector in self:
            yield key, vector


if __name__ == "__main__":
    pass
