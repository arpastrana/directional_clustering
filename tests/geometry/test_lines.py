#!/usr/bin/env python3

import os
import pytest

from directional_clustering.geometry import line_sdl
from directional_clustering.geometry import vector_lines_on_faces
from directional_clustering.geometry import line_tuple_to_dict

def test_line_sdl_OneSide(start, direction, length):
    """
    Tests if a line is create correctly with both sides = False.
    """
    ln = line_sdl(start, direction, length, False)
    assert ln == (start, [1.0, 1.0, 0.0])


def test_line_sdl_BothSides(start, direction, length):
    """
    Tests if a line is created correctly with both sides = True.
    """
    ln = line_sdl(start, direction, length, True)
    assert ln == ([-1.0, 1.0, 0.0], [1.0, 1.0, 0.0])

def test_lines_sdl_NoLength(start, direction):
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

def test_line_sd_l1D(direction, length):
    """
    Tests if the result will produce a 1D line if a 1D point is passed in.
    """
    start1D = [0.0]
    ln = line_sdl(start1D, direction, length)
    assert ln == ([-1.0], [1.0])

def test_vector_lines_on_faces_meshNoAttr(meshNoAttr, vector_tag):
    """
    Tests if the input of a mesh with no attributes raises an error.
    """
    with pytest.raises(ValueError):
        vector_lines_on_faces(meshNoAttr, vector_tag)

def test_vector_lines_on_faces_meshAttr(meshAttr, vector_tag):
    """
    Tests if the input of a mesh with attributes returns the correct line.
    """
    assert [([0.5,-0.5,0],[0.5,1.5,0])] == vector_lines_on_faces(meshAttr, vector_tag, factor=1)

def test_line_tuple_to_dict_1LinePassedIn(start):
    """
    Tests if the dictionary is created correctly if a line is passed in.
    """
    line = (start, [1,1,1])
    assert {'start': start, 'end': [1,1,1] } == line_tuple_to_dict(line)

def test_line_tuple_to_dict_1PointPassedIn(start):
    """
    Tests if a ValueError is raised if only one point is passed in.
    """
    with pytest.raises(ValueError):
        line_tuple_to_dict(start)

