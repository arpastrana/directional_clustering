# os
import os

# argument parsing helper
import fire

# file directories
from directional_clustering import RAWFIELD
from directional_clustering import JSON
from directional_clustering import OFF

# extended version of Mesh
from directional_clustering.mesh import MeshPlus
from directional_clustering.fields import NRoSyField


# ==============================================================================
# Main function
# ==============================================================================

def export_directional(filename, vf, degree=4, normalize=True):
    """
    Exports a mesh .OFF file and a vector field .rawfield.
    The goal is to interface with Directional's C++ field-to-mesh algorithm.
    """
# ==============================================================================
# Import mesh
# ==============================================================================

    json_in = os.path.abspath(os.path.join(JSON, filename + ".json"))
    mesh = MeshPlus.from_json(json_in)

# ==============================================================================
# Fetch vector_field from MeshPlus
# ==============================================================================

    vector_field = mesh.vector_field(vf)

# ==============================================================================
# Create NRoSy field
# ==============================================================================

    nrosy = NRoSyField.from_vector_field(vector_field, mesh, degree, normalize)

# ==============================================================================
# Export mesh as .OFF file
# ==============================================================================

    mesh.to_off(os.path.join(OFF, f"{filename}.off"))

# ==============================================================================
# Export .rawfield file
# ==============================================================================

    rawfield_out = os.path.abspath(os.path.join(RAWFIELD, f"{filename}_{vf}_{degree}rosy.rawfield"))
    nrosy.to_rawfield(rawfield_out)
    print(f"Exported rawfield to:\n{rawfield_out}")

# ==============================================================================
# Executable
# ==============================================================================

if __name__ == '__main__':
    fire.Fire(export_directional)
    print("Success!")
