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


if __name__ == "__main__":
    pass
