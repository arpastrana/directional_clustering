# os
import os

# argument parsing helper
import fire

# math
from math import radians

# COMPAS charm
from compas.geometry import Rotation
from compas.geometry import Vector

# file directories
from directional_clustering import JSON

# extended version of Mesh
from directional_clustering.mesh import MeshPlus

# ==============================================================================
# Main function
# ==============================================================================


def rotate_mesh(filename, vf, angle=15.0):
    """
    Rotates a mesh and one vector field around the global Z axis.
    """
# ==============================================================================
# Import mesh
# ==============================================================================

    json_in = os.path.abspath(os.path.join(JSON, filename + ".json"))
    mesh = MeshPlus.from_json(json_in)

# ==============================================================================
# Create rotation
# ==============================================================================

    axis = [0.0, 0.0, 1.0]
    rotation = Rotation.from_axis_and_angle(axis, radians(angle), point=[0.0, 0.0, 0.0])

# ==============================================================================
# Rotate mesh in-place
# ==============================================================================

    mesh.transform(rotation)

# ==============================================================================
# Rotate all vector fields
# ==============================================================================

    for vf in mesh.vector_fields():
        vector_field = mesh.vector_field(vf)

        for fkey in mesh.faces():
            vector = vector_field[fkey]
            vector = Vector(*vector)
            vector.transform(rotation)
            vector_field[fkey] = [vector.x, vector.y, vector.z]

        mesh.vector_field(vf, vector_field)

# ==============================================================================
# Export mesh as .JSON file
# ==============================================================================

    json_out = os.path.join(JSON, f"{filename}_rotated{angle}.json")
    mesh.to_json(json_out)
    print(f"Export mesh JSON file to: {json_out}")

# ==============================================================================
# Executable
# ==============================================================================

if __name__ == '__main__':
    fire.Fire(rotate_mesh)
    print("Success!")
