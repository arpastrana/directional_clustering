#!/usr/bin/env python3

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
def meshNoAttribute():
    """
    Mesh with 4 vertices and 1 face.
    """
    mesh = Mesh()

    a = mesh.add_vertex()  # x,y,z coordinates are optional and default to 0,0,0
    b = mesh.add_vertex(x=1)
    c = mesh.add_vertex(x=1, y=1)
    d = mesh.add_vertex(y=1)

    mesh.add_face([a, b, c, d])

    return mesh

@pytest.fixture
def meshAttribute(mesh):
    """
    Mesh with a vector as attribute. (WIP)
    """
    return mesh
