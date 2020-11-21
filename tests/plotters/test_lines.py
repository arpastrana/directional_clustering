import os
import pytest

from directional_clustering.plotters import line_sdl
from directional_clustering.plotters import vector_lines_on_faces
from directional_clustering.plotters import line_tuple_to_dict
from directional_clustering.plotters import polygon_list_to_dict

def test_line_sdl_one_side(start, direction, length):
    """
    Tests if a line is create correctly with both sides = False.
    """
    ln = line_sdl(start, direction, length, False)
    assert ln == (start, [1.0, 1.0, 0.0])


def test_line_sdl_both_sides(start, direction, length):
    """
    Tests if a line is created correctly with both sides = True.
    """
    ln = line_sdl(start, direction, length, True)
    assert ln == ([-1.0, 1.0, 0.0], [1.0, 1.0, 0.0])

def test_lines_sdl_no_length(start, direction):
    """
    Tests if a type error is raised when a line_sdl has no length as input.
    """
    with pytest.raises(TypeError):
        line_sdl(start, direction)

def test_line_sdl_2D(direction, length):
    """
    Tests if the result will produce a 2D line if a 2D point is passed in.
    """
    start2D = [0.0, 1.0]
    ln = line_sdl(start2D, direction, length)
    assert ln == ([-1.0, 1.0], [1.0, 1.0])

def test_line_sdl_1D(direction, length):
    """
    Tests if the result will produce a 1D line if a 1D point is passed in.
    """
    start1D = [0.0]
    ln = line_sdl(start1D, direction, length)
    assert ln == ([-1.0], [1.0])

def test_vector_lines_on_faces_mesh_no_attr(mesh_no_attr, vector_tag):
    """
    Tests if the input of a mesh with no attributes raises an error.
    """
    with pytest.raises(ValueError):
        vector_lines_on_faces(mesh_no_attr, vector_tag)

def test_vector_lines_on_faces_mesh_attr(mesh_attr, vector_tag):
    """
    Tests if the input of a mesh with attributes returns the correct line.
    """
    assert [([0.5, -0.5, 0.0],[0.5, 1.5, 0.0])] == vector_lines_on_faces(mesh_attr, vector_tag, factor=1)

def test_line_tuple_to_dict_1_line_passed_in(start):
    """
    Tests if the dictionary is created correctly if a line is passed in.
    """
    line = (start, [1.0, 1.0, 1.0])
    assert {'start': start, 'end': [1.0, 1.0, 1.0]} == line_tuple_to_dict(line)

def test_line_tuple_to_dict_1_point_passed_in(start):
    """
    Tests if a ValueError is raised if only one point is passed in.
    """
    with pytest.raises(ValueError):
        line_tuple_to_dict(start)

def test_polygon_list_to_dict_one_point():
    """
    Tests if a dictionary is returned when one point is passed in.
    """
    assert {'points': [1.0, 0.0, 0.0]} == polygon_list_to_dict([1.0, 0.0, 0.0])

def test_polygon_list_to_dict_empty():
    """
    Tests if TypeError is raised when nothing is passed in.
    """
    with pytest.raises(TypeError):
        polygon_list_to_dict()

