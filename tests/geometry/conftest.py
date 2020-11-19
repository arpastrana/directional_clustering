#!/usr/bin/env python3

import pytest


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
