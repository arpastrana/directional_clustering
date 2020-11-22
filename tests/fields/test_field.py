import pytest

from directional_clustering.fields import Field


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def field_3d():
    """
    A field with a dimensionality of three with 2 entries.
    """
    field = Field(3)
    field[0] = [0.0, 1.0, 2.0]  # vector_3d
    field[1] = [1.0, 1.0, 1.0]
    return field


@pytest.fixture
def field_2d():
    """
    A field with a dimensionality of two.
    """
    return Field(2)


@pytest.fixture
def vector_3d():
    """
    A vector with 3 entries.
    """
    return [0.0, 1.0, 2.0]


@pytest.fixture
def vector_2d():
    """
    A vector with 2 entries.
    """
    return [0.0, 1.0]


# ==============================================================================
# Tests
# ==============================================================================


def test_field_size(field_2d, field_3d):
    """
    Checks the number of entries in a field is correct.
    """
    assert field_2d.size() == 0
    assert field_3d.size() == 2


def test_field_dimensionality(field_2d, field_3d):
    """
    Checks the number of dimensions in a field.
    """
    assert field_2d.dimensionality() == 2
    assert field_3d.dimensionality() == 3


def test_field_get_item(field_3d, vector_3d):
    """
    Checks that the right item is queried by key.
    """
    assert field_3d[0] == vector_3d


def test_field_get_wrong_item(field_3d):
    """
    Key won't be found in field.
    """
    with pytest.raises(KeyError):
        field_3d[999]


def test_field_set_item(field_2d, vector_2d):
    """
    Store an item and query it.
    """
    field_2d[0] = vector_2d[:]
    assert field_2d[0] == vector_2d


def test_field_set_wrong_item(field_2d, vector_3d):
    """
    Storing a 3d vector in a 2d field.
    """
    with pytest.raises(ValueError):
        field_2d[0] = vector_3d


def test_field_delete_item(field_3d):
    """
    Removes an entry in the field and queries it afterwards. Should fail.
    """
    del field_3d[0]
    with pytest.raises(KeyError):
        field_3d[0]


def test_field_delete_wrong_item(field_3d):
    """
    Removes an entry in the field and queries it afterwards. Should fail.
    """
    with pytest.raises(KeyError):
        del field_3d[999]


def test_field_iter(field_3d, vector_3d):
    """
    Ensures iterators goes over all entries in the field.
    """
    counter = 0
    for key, item in field_3d:
        counter += 1

    assert counter == field_3d.size()
