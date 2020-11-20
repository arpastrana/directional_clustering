#!/usr/bin/env python3

import os
import pytest

from directional_clustering.geometry import line_sdl
from directional_clustering.geometry import vector_lines_on_faces


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

def test_vector_lines_on_faces_(): #(WIP)
    pass
