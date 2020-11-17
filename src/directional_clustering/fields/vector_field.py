#!/usr/bin/env python3

"""
Create a vector field.
"""

from directional_clustering.fields import Field


class VectorField(Field):
    """
    A field with a dimensionality of 3.
    A mapping from keys to vectors.
    """
    def __init__(self):
        """
        """
        dimensionality = 3
        super(VectorField, self).__init__(dimensionality)

    def add_vector(self, key, vector):
        """
        Adds a vector to a vector field.
        """
        self[key] = vector

    def remove_vector(self, key):
        """
        Deletes a vector from the field.
        """
        del self[key]

    def vector(self, key):
        """
        Get a vector from a vector field.
        """
        return self[key]

    def keys(self):
        """
        Iterate over they keys of the vector field.
        """
        for key, _ in self:
            yield key

    def vectors(self):
        """
        Iterate over the vectors of the vector field.
        """
        for _, vector in self:
            yield vector

    @classmethod
    def from_sequence(cls, sequence):
        """
        Create a vector field from a sequence.
        """
        vf = cls()

        for index, vector in enumerate(sequence):
            vf.add_vector(index, vector)
        return cls()

    def to_sequence(self):
        """
        Converts a vector field into a numpy array.
        """
        vectors = []
        for key in self.keys():
            vectors.append(self.vector(key))
        return vectors

    def index_key(self):
        """
        Makes an index to keys mapping.
        """
        return {index: key for index, key in enumerate(self.keys())}

    def key_index(self):
        """
        Makes a keys to index mapping.
        """
        return {key: index for index, key in self.index_key()}


if __name__ == "__main__":

    vf = VectorField()

    print(vf.dimensionality())
    print(vf.size())

    vf.add_vector(0, [0, 0, 1])

    try:
        vf.add_vector(1, [0, 0])
    except ValueError:
        pass
    else:
        raise "Something went wrong!"

    vf.add_vector(1, [1, 0, 0])

    print("Vector at 0", vf.vector(0))
    print("Vector field keys", list(vf.keys()))
    print("Vector field vectors", list(vf.vectors()))

    vf.remove_vector(0)
    vf.remove_vector(1)

    assert vf.size() == 0
