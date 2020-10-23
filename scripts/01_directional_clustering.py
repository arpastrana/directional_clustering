# =============================================================================
# 01. Define parameters
# =============================================================================

# Just for reference, this is a list with all available vector fields
# which have been stored in the json file.
# First and second principal directions are always orthogonal to each other.

vector_field_names= [
    "n_1",  # axial forces in first principal direction
    "n_2",  # axial forces in second principal direction
    "m_1",  # bending moments in first principal direction
    "m_2",  # bending moments in second principal direction
    "ps_1_top",  # first principal direction stress direction at topmost fiber
    "ps_1_bot",  # first principal direction stress direction at bottommost fiber
    "ps_1_mid",  # first principal direction stress direction at middle fiber
    "ps_2_top",  # second principal direction stress direction at topmost fiber
    "ps_2_bot",  # second principal direction stress direction at bottommost fiber
    "ps_2_mid",  # second principal direction stress direction at middle fiber
    "custom_1",  # an arbitrary vector field pointing in the global X direction
    "custom_2"   # an arbitrary vector field pointing in the global X direction
    ]



JSON_IN = "../data/json_files/perimeter_supported_slab"  # schlaich

tag = "m_1"
