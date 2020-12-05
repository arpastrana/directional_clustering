import pytest

from numpy import array
from numpy import all


from directional_clustering.plotters import mesh_to_vertices_xyz
from directional_clustering.plotters import trimesh_face_connect
from directional_clustering.plotters import lines_to_start_end_xyz
from directional_clustering.plotters import lines_xyz_to_tables
from directional_clustering.plotters import lines_start_end_connect
from directional_clustering.plotters import vectors_dict_to_array
from directional_clustering.plotters import face_centroids


def test_mesh_to_vertices_xyz(trimesh_attr):
    """
    Tests if each coordinate list is organized correctly.
    """
    x, y, z = mesh_to_vertices_xyz(trimesh_attr)
    check_x = [0.0, 1.0, 1.0]
    check_y = [0.0, 0.0, 1.0]
    check_z = [0.0, 0.0, 0.0]

    assert all([[x == check_x],[y == check_y], [z == check_z]])


def test_trimesh_face_connect(trimesh_attr):
    """
    Tests if each coordinate list is organized correctly.
    """
    i, j, k = trimesh_face_connect(trimesh_attr)

    assert all([[i == [0]], [j == [1]], [k == [2]]])
