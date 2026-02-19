import pytest

import numpy as np

from compas.datastructures import Mesh

from directional_clustering.fields import VectorField

from directional_clustering.clustering import KMeans
from directional_clustering.clustering import CosineKMeans
from directional_clustering.clustering import VariationalKMeans
from directional_clustering.clustering import DifferentiableCosineKMeans


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def random_array():
    """
    A 2x2 array with random float values.
    """
    return np.random.rand(2, 2)


@pytest.fixture
def cosine_array():
    """
    A 3x2 array with float values.
    """
    return np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


@pytest.fixture
def cosine_centroids():
    """
    A 2x2 array of floats that represents the centers of two 2D clusters.
    """
    return np.array([[1.0, 0.0], [1.0, 1.0]])


@pytest.fixture
def vectors():
    """
    A list with three 3d vectors.
    """
    return [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 2.0, 0.0]]


@pytest.fixture
def seeds():
    """
    A list with two 3d vectors.
    """
    return [[0.0, 0.0, 1.0], [0.0, 2.0, 0.0]]


@pytest.fixture
def seed_array(seeds):
    """
    A numpy array with two 3d vectors.
    """
    return np.array(seeds)


@pytest.fixture
def vector_array(vectors):
    """
    A numpy array with three 3d vectors.
    """
    return np.array(vectors)


@pytest.fixture
def mesh(vectors):
    """
    A COMPAS mesh with three vectors stored as face attributes.
    """
    _mesh = Mesh()

    # add vertices
    for i in range(5):
        _mesh.add_vertex(key=i, x=i, y=i, z=i)

    # right-hand side winding -- normals pointing up
    _mesh.add_face(fkey=0, vertices=[0, 1, 2])
    _mesh.add_face(fkey=1, vertices=[0, 2, 3])
    _mesh.add_face(fkey=2, vertices=[0, 3, 4])

    name = "my_vector_field"
    _mesh.face_attribute(key=0, name=name, value=vectors[0])
    _mesh.face_attribute(key=1, name=name, value=vectors[1])
    _mesh.face_attribute(key=2, name=name, value=vectors[2])

    return _mesh


@pytest.fixture
def vector_field(vectors):
    """
    A vector field with three vectors.
    """
    vf = VectorField()

    for key, vector in enumerate(vectors):
        vf.add_vector(key, vector)

    return vf


@pytest.fixture
def n_clusters():
    """
    The number of clusters to make.
    """
    return 2


@pytest.fixture
def iters():
    """
    The number of iterations to run the clustering algorithms for.
    """
    return 100


@pytest.fixture
def tol():
    """
    The convergence tolerance.
    """
    return 1e-6


@pytest.fixture
def kmeans(mesh, vector_field):
    """
    An instance of KMeans.
    """
    return KMeans(mesh, vector_field)


@pytest.fixture
def cosine_kmeans(mesh, vector_field):
    """
    An instance of CosineKMeans.
    """
    return CosineKMeans(mesh, vector_field)


@pytest.fixture
def variational_kmeans(mesh, vector_field):
    """
    An instance of VariationalKMeans.
    """
    return VariationalKMeans(mesh, vector_field)


@pytest.fixture
def diff_cosine_kmeans(mesh, vector_field):
    """
    An instance of DifferentiableCosineKMeans.
    """
    return DifferentiableCosineKMeans(mesh, vector_field)
