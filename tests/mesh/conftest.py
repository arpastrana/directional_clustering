import os

import pytest

from compas.datastructures import Mesh
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
