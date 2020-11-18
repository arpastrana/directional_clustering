#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import directional_clustering
import pytest
from directional_clustering.geometry import line_sdl

def test_dummy_fails():
    string = "hello_world"
    assert string != "hello"


def test_dummy_success():
    string = "hello_world"
    assert string == "hello_world"

def test_line_sdl():
    start = [0.0, 1.0, 0.0]
    direction = [1.0, 0.0, 0.0]
    length = 1.0
    ln = line_sdl(start, direction, length, False)
    assert ln == (start, [1.0, 1.0, 0.0])
