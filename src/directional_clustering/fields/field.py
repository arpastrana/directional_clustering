from directional_clustering.fields import AbstractField


__all__ = ["Field"]


class Field(AbstractField):
    """
    A concrete field.

    Basically, a container for scalars and vectors.
    One key can store exclusively one value at a time.

    It is crucial to have it as a datastructure where a field's entries
    are accessed with the keys of the Mesh they are coupled to.

    Parameters
    ----------
    dimensionality : `int`
        The dimensionality of the field.
    """
    def __init__(self, dimensionality):
        """
        The constructor.
        """
        assert isinstance(dimensionality, int)

        self._dimensionality = dimensionality
        self._size = None
        self._field = dict()

    def __getitem__(self, key):
        """
        Retrieves a field value by key.

        Parameters
        ----------
        key : `int`
            An access key representing a pointer to a mesh entity.

        Returns
        -------
        item : `list`
            The queried item in the field.
        """
        return self._field[key]

    def __setitem__(self, key, item):
        """
        Sets an entry in the field.

        Parameters
        ----------
        key : `int`
            An access key representing a pointer to a mesh entity.
        item : `list`
            The item to store.
        """
        if len(item) != self.dimensionality():
            msg = "Length of {} is incompatible with the field's dimensionality"
            raise ValueError(msg.format(item))

        self._field[int(key)] = item

    def __delitem__(self, key):
        """
        Deletes a key and dereferences the item it points to.

        Parameter
        ---------
        key : `int`
            The access key of the item to remove.
        """
        del self._field[key]

    def __iter__(self):
        """
        Iterates over the keys and items of the field.

        Yields
        ------
        key, item : `tuple`
            The access key and the item it points to.
        """
        for key, item in self._field.items():
            yield key, item

    def dimensionality(self):
        """
        The fixed dimensionality of a field.

        Returns
        -------
        dimensionality : `int`
            The dimensionality of the field.
        """
        return self._dimensionality

    def size(self):
        """
        The number of items stored in the field.

        Returns
        -------
        size : `int`
            The number of items.
        """
        return len(self._field)

    # --------------------------------------------------------------------------
    # IO
    # --------------------------------------------------------------------------

    def to_sequence(self):
        """
        Converts the field into a sequence.

        Returns
        -------
        sequence : `list` of `list`
            A list of vectors.

        Notes
        -----
        The output vectors are not sorted by their access keys.
        """
        return [value for _, value in self]

    @classmethod
    def from_sequence(cls, sequence):
        """
        Creates a field from a sequence.

        Parameters
        ----------
        sequence : `list` of `list`
            A list of vectors.

        Returns
        -------
        field : `directional_clustering.fields.Field`
            A field.

        Notes
        -----
        The input vectors are stored in the order they are supplied.
        Access keys are generated in the range from 0 to the sequence length.
        """
        vf = cls()

        for index, vector in enumerate(sequence):
            vf[index] = vector

        return vf

    @classmethod
    def from_mesh(cls, mesh, name):
        """
        Extracts a field from the faces of a mesh.

        Parameters
        ----------
        mesh : `directional_clustering.mesh.MeshPlus`
            A mesh.
        name : `str`
            The name of the face attribute to query.

        Returns
        -------
        field : `Field`
            A field.

        Notes
        -----
        Every item in the field is stored with the mesh face keys as access keys.
        """
        field = cls()

        for fkey in mesh.faces():
            item = mesh.face_attribute(fkey, name)

            msg = "Attribute {} is not defined on face {}!".format(name, fkey)
            assert item is not None, msg

            field[fkey] = item

        return field


if __name__ == "__main__":
    pass
