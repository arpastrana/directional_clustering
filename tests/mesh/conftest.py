import os

import pytest

from directional_clustering.mesh import MeshPlus
from directional_clustering.fields import VectorField
from directional_clustering import JSON


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def my_mesh():
    """
    A self-maded MeshPlus mesh with two faces
    """
    _mesh = MeshPlus()

    # add vertices
    for i in range(4):
        _mesh.add_vertex(key=i)

    # right-hand side winding -- normals pointing up
    _mesh.add_face(fkey=0, vertices=[0, 1, 2])
    _mesh.add_face(fkey=1, vertices=[0, 2, 3])

    return _mesh

@pytest.fixture
def my_vector_field():
    """
    A vector field compatibale with the mesh above.
    """
    field = VectorField()

    field.add_vector(0, [0.0, 1.0, 2.0])
    field.add_vector(1, [1.0, 1.0, 1.0])

    return field

@pytest.fixture
def mesh():
    """
    A MeshPlus imported from JSON file
    """
    name_in = "perimeter_supported_slab.json"
    JSON_IN = os.path.abspath(os.path.join(JSON, name_in))

    _mesh = MeshPlus.from_json(JSON_IN)

    return _mesh




