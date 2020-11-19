#!/usr/bin/env python3

import pytest

from directional_clustering.transformations import adjacent_vectors
from directional_clustering.transformations import mean_vector
from directional_clustering.transformations import smoothed_vector
from directional_clustering.transformations import smoothen_vector_field


# ==============================================================================
# Tests
# ==============================================================================


def test_mean_vector_empty(empty_list):
    """
    Check if an empty least is passed to mean_vector.
    """
    with pytest.raises(ValueError):
        mean_vector(empty_list)


def test_mean_vector_redudant(redundant_list):
    """
    Should return any of the entries.
    """
    assert mean_vector(redundant_list) == redundant_list[0]


def test_mean_vector(vectors_list):
    """
    Regular behavior.
    """
    assert mean_vector(vectors_list) == [1.5, 0.0, 1.5]


def test_adjacent_vectors(vector_field, adjacency):
    """
    Regular behavior
    """
    vectors = [[1.0, -1.0, 0.0], [1.0, 1.0, 2.0]]
    assert adjacent_vectors(vector_field, adjacency[0]) == vectors


def test_adjacent_vectors_not_found(vector_field, flawed_adjacency):
    """
    Adjacency is wrong.
    """
    with pytest.raises(KeyError):
        for key in vector_field.keys():
            adjacent_vectors(vector_field, flawed_adjacency[key])


@pytest.mark.parametrize("damping", [-1.0, 10.0])
def test_smoothed_vector_wrong_damping(vector_single, vector_smoothed, damping):
    """
    Damping values must be between 0 and 1.0.
    """
    with pytest.raises(AssertionError):
        smoothed_vector(vector_single, vector_smoothed, damping)


@pytest.mark.parametrize("damping", [0.1 * _ for _ in range(10)])
def test_smoothed_vector_returns_same(vector_single, damping):
    """
    Smoothing unaffects input vector
    """
    assert smoothed_vector(vector_single, vector_single, damping) == vector_single


def test_smoothen_vector_field_wrong_size(vector_field, adjacency):
    """
    Adjacency has less entries than there are in the vector field.
    """
    with pytest.raises(AssertionError):
        smoothen_vector_field(vector_field, adjacency, 1)


def test_smoothen_vector_field_solitary_faces(vector_field, adjacency_full):
    """
    Adjacency is complete but some faces have no neighbors.
    They should remain unchanged.
    """
    ref_vectors = {key: vector for key, vector in vector_field.items()}

    smoothen_vector_field(vector_field, adjacency_full, iters=1, damping=0.5)

    for key, value in adjacency_full.items():
        if not value:
            assert vector_field.vector(key) == ref_vectors[key]


@pytest.mark.parametrize("damping", [0.1 * _ for _ in range(10)])
def test_smoothen_vector_field(vector_field, adjacency_full, damping):
    """
    Regular behavior for first entry.
    """
    smoothen_vector_field(vector_field, adjacency_full, iters=1, damping=damping)
    vector = vector_field.vector(0)
    assert vector == [1.0, 0.0, 1.0 - damping]
