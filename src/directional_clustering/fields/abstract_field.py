#!/usr/bin/env python3

"""
An abstract implementation of a Field defined on a Mesh.

It is crucial to have it as a datastructure where fields' entries
are accessed with the keys of the Mesh they are coupled to.

If this was a dictionary, and the vector field lives on the faces
of a mesh, the way to access entry i in the vector field would be

entry = Field[face_key]
entry = Field[field entry]
"""

from abc import ABC
from abc import abstractmethod


__all__ = ["AbstractField", "Field"]


class AbstractField(ABC):
    """
    An abstract field.
    """
    @abstractmethod
    def dimensionality(self):
        """
        The fixed dimensionality of a field.
        """
        return self._dimensionality

    @abstractmethod
    def size(self):
        """
        The number of entries in the field.
        """
        return


class Field(AbstractField):
    """
    An abstract field.
    Basically, a container for scalars and vectors.

    One key can store exclusively one value at a time.
    """
    def __init__(self, dimensionality):
        """
        """
        self._dimensionality = dimensionality
        self._size = None
        self._field = dict()

    def __getitem__(self, key):
        """
        Retrieves a field value by key.
        """
        return self._field[key]

    def __setitem__(self, key, value):
        """
        Sets the value referenced by a key.
        """
        if len(value) != self.dimensionality():
            msg = "Length of {} is incompatible with the field's dimensionality"
            raise ValueError(msg.format(value))

        self._field[key] = value

    def __delitem__(self, key):
        """
        Removes a key and the value it references.
        """
        del self._field[key]

    def __iter__(self):
        """
        Iterates over the keys and values of the field.
        """
        for key, value in self._field.items():
            yield key, value

    def dimensionality(self):
        """
        The fixed dimensionality of a field.
        """
        return self._dimensionality

    def size(self):
        """
        The number of entries in the field.
        """
        return len(self._field)


if __name__ == "__main__":
    # small tests

    field = Field(2)
    try:
        field[25] = [0, 1, 2]
    except ValueError:
        field[25] = [0, 1]

    assert len(list(field)) == field.size()
    assert field.dimensionality() == 2

    assert field[25] == [0, 1]

    del field[25]

    try:
        a = field[25]
    except KeyError:
        pass
