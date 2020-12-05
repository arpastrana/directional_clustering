import pytest

from directional_clustering.fields import VectorField


# ==============================================================================
# Tests
# ==============================================================================


def test_vector_field_dimensionality(vector_field):
    """
    Checks that the dimensionality is equal to three.
    """
    assert vector_field.dimensionality() == 3


def test_vector_field_size(vector_field):
    """
    Asserts that the number of vectors in the field is two.
    """
    assert vector_field.size() == 2


def test_vector_field_add_wrong_entry(vector_field, vector_2d):
    """
    Fails because vector is 2d and vector field only takes vectors in 3d.
    """
    with pytest.raises(ValueError):
        vector_field.add_vector(0, vector_2d)


def test_vector_field_add(vector_field, vector_3d):
    """
    Ensures vector was stored with the right key.
    """
    vector_field.add_vector(2, vector_3d)
    assert vector_field.vector(2) == vector_3d


def test_vector_field_remove_all(vector_field):
    """
    Tests that a vector field is empty after removing all vectors.
    """
    keys = list(vector_field.keys())
    for key in keys:
        vector_field.remove_vector(key)

    assert vector_field.size() == 0


def test_vector_field_remove_wrong_key(vector_field):
    """
    Attempts to remove an unexistant entry from a vector field.
    """
    with pytest.raises(KeyError):
        vector_field.remove_vector(999)


def test_vector_field_keys(vector_field, field_keys):
    """
    Checks that the iterator returns all the keys in the vector field.
    """
    counter = 0
    for key in vector_field.keys():
        counter += 1
        assert key in field_keys

    assert counter == len(field_keys)


def test_vector_field_vectors(vector_field, vectors_3d):
    """
    Checks that the iterator returns all the vectors in the vector field.
    """
    counter = 0
    for vector in vector_field.vectors():
        counter += 1
        assert vector in vectors_3d

    assert counter == len(vectors_3d)


def test_vector_field_to_sequence(vector_field):
    """
    Tests that a list of length 2 is output.
    """
    seq = vector_field.to_sequence()
    assert len(seq) == vector_field.size()
    assert isinstance(seq, list)


def test_vector_field_from_sequence(vectors_3d):
    """
    Checks that vectors are added in the order of the sequence.
    """
    vector_field = VectorField.from_sequence(vectors_3d)
    assert vector_field.vector(0) == vectors_3d[0], vector_field.keys()
    assert vector_field.vector(1) == vectors_3d[1]
    assert vector_field.size() == len(vectors_3d)


def test_vector_field_from_mesh_faces(mesh):
    """
    Adds all the vectors stored as face attributes in a mesh.
    """
    vector_field = VectorField.from_mesh_faces(mesh, "my_vector_field")
    assert vector_field.size() == mesh.number_of_faces()


def test_vector_field_from_mesh_faces_fails(mesh):
    """
    Attempts to create a vector field from an unexisting attribute.
    """
    with pytest.raises(AssertionError):
        VectorField.from_mesh_faces(mesh, "unexisting_field")
