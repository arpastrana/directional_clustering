#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import directional_clustering
import pytest
from directional_clustering.geometry import line_sdl


def test_line_sdlOneSide():
    start = [0.0, 1.0, 0.0]
    direction = [1.0, 0.0, 0.0]
    length = 1.0
    ln = line_sdl(start, direction, length, False)
    assert ln == (start, [1.0, 1.0, 0.0])


def test_line_sdlBothSides():
    start = [0.0, 1.0, 0.0]
    direction = [1.0, 0.0, 0.0]
    length = 1.0
    ln = line_sdl(start, direction, length, True)
    assert ln == ([-1.0, 1.0, 0.0], [1.0, 1.0, 0.0])
