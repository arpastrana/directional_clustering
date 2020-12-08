import pytest


# ==============================================================================
# Tests
# ==============================================================================

def test_meshplus_vector_field(my_mesh, my_vector_field):
    """
    Checks the right vaector is stored and get at the same time
    """
    name = "my_vector_field"
    my_mesh.vector_field(name, my_vector_field)
    field_extracted = my_mesh.vector_field(name)
    
    assert field_extracted[0] == my_vector_field[0]
    assert field_extracted[1] == my_vector_field[1]

def test_meshplus_vector_fields(mesh):
    """
    Checks all vector field attributes of a mesh are searched out
    """
    all_attributes = ["n_1", "n_2", "m_1", "m_2", "ps_1_top", "ps_1_bot", "ps_1_mid",  
                      "ps_2_top", "ps_2_bot", "ps_2_mid", "custom_1", "custom_2"]
    detected_attributes = mesh.vector_fields()
    
    assert len(set(all_attributes).difference(set(detected_attributes))) == 0
