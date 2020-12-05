import pytest

from compas.datastructures import Mesh


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def start():
    """
    3D point as start point.
    """
    return [0.0, 1.0, 0.0]

@pytest.fixture
def end():
    """
    3D point as end point.
    """
    return [1.0, 1.0, 0.0]

@pytest.fixture
def direction():
    """
    3D vector as direction.
    """
    return [1.0, 0.0, 0.0]

@pytest.fixture
def length():
    """
    Length.
    """
    return 1.0

@pytest.fixture
def vector_tag():
    """
    A string to tag a vector.
    """
    return "a_vector"

@pytest.fixture
def quadmesh_no_attr():
    """
    Mesh with 4 vertices and 1 face.
    """
    mesh = Mesh()
    a = mesh.add_vertex(x=0, y=0, z=0)
    b = mesh.add_vertex(x=1, y=0, z=0)
    c = mesh.add_vertex(x=1, y=1, z=0)
    d = mesh.add_vertex(x=0, y=1, z=0)
    mesh.add_face([a, b, c, d])

    return mesh

@pytest.fixture
def mesh_attr(quadmesh_no_attr, vector_tag):
    """
    Mesh with a vector as attribute.
    """
    faces = list(quadmesh_no_attr.faces())
    quadmesh_no_attr.faces_attribute(vector_tag, [0, 1, 0], faces)
    return quadmesh_no_attr

@pytest.fixture
def trimesh_attr(vector_tag):
    """
    Mesh with 3 vertices and 1 face.
    """
    mesh = Mesh()
    a = mesh.add_vertex(x=0.0, y=0.0, z=0.0)
    b = mesh.add_vertex(x=1.0, y=0.0, z=0.0)
    c = mesh.add_vertex(x=1.0, y=1.0, z=0.0)
    mesh.add_face([a, b, c])

    faces = list(mesh.faces())
    mesh.faces_attribute(vector_tag, [0, 1, 0], faces)
    return mesh
