import pytest

from compas.datastructures import Mesh

from directional_clustering.fields import Field
from directional_clustering.fields import VectorField


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
    A field with a dimensionality of 2.
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


@pytest.fixture
def vectors_3d():
    """
    A list with 2 vectors.
    """
    return [[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]]


@pytest.fixture
def vector_field():
    """
    A vector field with two vectors.
    """
    field = VectorField()

    field.add_vector(0, [0.0, 1.0, 2.0])
    field.add_vector(1, [1.0, 1.0, 1.0])

    return field


@pytest.fixture
def field_keys():
    """
    Two integer keys.
    """
    return [0, 1]


@pytest.fixture
def mesh():
    """
    A COMPAS mesh with two vectors stored as face attributes.
    """
    _mesh = Mesh()

    # add vertices
    for i in range(4):
        _mesh.add_vertex(key=i)

    # right-hand side winding -- normals pointing up
    _mesh.add_face(fkey=0, vertices=[0, 1, 2])
    _mesh.add_face(fkey=1, vertices=[0, 2, 3])

    name = "my_vector_field"
    _mesh.face_attribute(key=0, name=name, value=[0.0, 0.0, 1.0])
    _mesh.face_attribute(key=1, name=name, value=[0.0, 0.0, 2.0])

    return _mesh
