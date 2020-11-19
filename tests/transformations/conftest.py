#!/usr/bin/env python3

import pytest

from directional_clustering.fields import VectorField


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def vector_field():
    """
    """
    vectors = {0: [1.0, 0.0, 0.0],
               1: [1.0, -1.0, 0.0],
               2: [1.0, 1.0, 2.0]}

    v_field = VectorField()

    for key, vector in vectors.items():
        v_field.add_vector(key, vector)

    return v_field


@pytest.fixture
def adjacency():
    """
    """
    return {0: [1, 2]}

@pytest.fixture
def adjacency_full():
    """
    """
    return {0: [1, 2], 1: [], 2: []}

@pytest.fixture
def empty_list():
    """
    """
    return []


@pytest.fixture
def flawed_adjacency():
    """
    """
    return {9: [1, 2]}


@pytest.fixture
def redundant_list():
    """
    """
    return [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]


@pytest.fixture
def vectors_list():
    """
    """
    return [[1.0, 0.0, 1.0], [2.0, 0.0, 2.0]]


@pytest.fixture
def vector_smoothed():
    """
    """
    return [1.0, 0.0, 0.0]


@pytest.fixture
def vector_single():
    """
    """
    return [1.0, 0.0, 1.0]
