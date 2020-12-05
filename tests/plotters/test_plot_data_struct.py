import pytest

from numpy import array
from numpy import all
from numpy import isnan

from directional_clustering.plotters import mesh_to_vertices_xyz
from directional_clustering.plotters import trimesh_face_connect
from directional_clustering.plotters import lines_to_start_end_xyz
from directional_clustering.plotters import lines_xyz_to_tables
from directional_clustering.plotters import coord_start_end_none
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

    assert all([[x==check_x], [y==check_y], [z==check_z]])


def test_trimesh_face_connect(trimesh_attr):
    """
    Tests output of lists for each face.
    """
    i, j, k = trimesh_face_connect(trimesh_attr)

    assert all([[i==[0]], [j==[1]], [k==[2]]])

def test_lines_to_start_end_xyz(start, end):
    """
    Test order and type (list) of output.
    """
    ln = (start, end)
    sx, sy, sz, ex, ey, ez = lines_to_start_end_xyz([ln])

    assert all([[sx==start[0]], [sy==start[1]], [sz==start[2]],
                [ex==end[0]], [ey==end[1]], [ez==end[2]]])

def test_lines_xyz_to_tables(start, end):
    """
    Test order of 2D list output.
    """
    tx, ty, tz = lines_xyz_to_tables([start[0]], [start[1]], [start[2]], [end[0]], [end[1]], [end[2]])

    assert all([start[0]==tx[0][0], end[0]==tx[0][1],
                start[1]==ty[0][0], end[1]==ty[0][1],
                start[2]==tz[0][0], end[2]==tz[0][1]])

def test_coord_start_end_none():
    """
    Test order and types list output.
    """
    nums_1st = [0.0, 1.0]
    nums_2nd = [2.0, 3.0]
    num_lines = 2
    c = coord_start_end_none(nums_1st, nums_2nd, num_lines)
    assert all([0.0==c[0], 2.0==c[1], isnan(c[2]),
                1.0==c[3], 3.0==c[4], isnan(c[5])])

def test_lines_start_end_connect():
    """
    Test
    """
    pass

def test_vectors_dict_to_array():
    """
    Test
    """
    pass

def test_face_centroids():
    """
    Test
    """
    pass
