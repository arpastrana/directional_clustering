from directional_clustering.fields import Field


__all__ = ["VectorField"]


class VectorField(Field):
    """
    A field with a fixed dimensionality of 3.
    """
    def __init__(self):
        """
        The constructor.
        """
        dimensionality = 3
        super(VectorField, self).__init__(dimensionality)

    # --------------------------------------------------------------------------
    # Vector operations
    # --------------------------------------------------------------------------

    def add_vector(self, key, vector):
        """
        Adds a vector entry to a vector field.

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
        Deletes a vector from the vector field.

        Parameters
        ----------
        key : `int`
            The key of the vector to remove.
        """
        del self[key]

    def vector(self, key):
        """
        Queries a vector from a vector field.

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
        Iterates over they access keys of the vector field.

        Yields
        ------
        key : `int`
            The next access key in the vector field.
        """
        for key, _ in self:
            yield key

    def vectors(self):
        """
        Iterates over the vectors of the vector field.

        Yields
        ------
        vector : `list` of `float`
            The next vector in the vector field.
        """
        for _, vector in self:
            yield vector

    def items(self):
        """
        Iterates over the keys and the vectors of the field.

        Yields
        ------
        key : `int`
            The next access key in the vector field.
        vector : `list` of `float`
            The next vector in the vector field.
        """
        for key, vector in self:
            yield key, vector

    # --------------------------------------------------------------------------
    # IO
    # --------------------------------------------------------------------------

    def to_sequence(self):
        """
        Converts a vector field into a sequence.

        Returns
        -------
        sequence : `list` of `list`
            A list of vectors.

        Notes
        -----
        The output vectors are not sorted by their access keys.
        """
        return [vector for vector in self.vectors()]

    @classmethod
    def from_sequence(cls, sequence):
        """
        Creates a vector field from a sequence.

        Parameters
        ----------
        sequence : `list` of `list`
            A list of vectors.

        Returns
        -------
        vector_field : `directional_clustering.fields.VectorField`
            A vector field.

        Notes
        -----
        The vectors are stored in the order they are supplied.
        Access keys are generated in the range from 0 to the sequence length.
        """
        vf = cls()

        for index, vector in enumerate(sequence):
            vf.add_vector(index, vector)

        return vf

    @classmethod
    def from_mesh_faces(cls, mesh, name):
        """
        Extracts a vector field from the faces of a mesh.

        Parameters
        ----------
        mesh : `directional_clustering.mesh.MeshPlus`
            A mesh.
        name : `str`
            The name of the face attribute to query.

        Returns
        -------
        vector_field : `VectorField`
            A vector field.

        Notes
        -----
        Deprecated.
        Every vector is stored with the mesh face keys as access keys.
        """
        vector_field = cls()

        for fkey in mesh.faces():
            vector = mesh.face_attribute(fkey, name)

            msg = "Attribute {} is not defined on face {}!".format(name, fkey)
            assert vector is not None, msg

            vector_field.add_vector(fkey, vector)

        return vector_field


if __name__ == "__main__":
    pass
