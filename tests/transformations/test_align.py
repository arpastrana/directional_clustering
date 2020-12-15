import pytest

from compas.geometry import scale_vector

from directional_clustering.transformations import align_vector_field


# ==============================================================================
# Tests
# ==============================================================================

@pytest.mark.parametrize("reference,sign", [([1.0, 0.0, 0.0], 1.0),
                                            ([-1.0, 0.0, 0.0], -1.0)])
def test_align_vector_field(vector_field, reference, sign):
    """
    Tests vector field alignment to a reference vector.
    """
    vectors = {key: vector for key, vector in vector_field.items()}
    align_vector_field(vector_field, reference)

    for key, vector in vector_field.items():
        assert vector == scale_vector(vectors[key], sign), vector
