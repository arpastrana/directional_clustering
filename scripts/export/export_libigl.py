# os
import os

# argument parsing helper
import fire

# hello numpy, my old friend
import numpy as np

# compas and friends
from compas.geometry import normalize_vector

# Library file directories
from directional_clustering import JSON

# extended version of Mesh
from directional_clustering.mesh import MeshPlus


# ==============================================================================
# Globals
# ==============================================================================

THERE = "/Users/arpj/code/libraries/libigl/tutorial/508_ARP_MIQ/"

# ==============================================================================
# Main function
# ==============================================================================

def export_libigl(filename, vf_name="m_1_k", vf_name_90="m_2_k"):
    """
    Export text files to interact with Libigl's C++ MIQ algorithm.
    """

# ==============================================================================
# Import mesh
# ==============================================================================

    name_in = filename + ".json"
    json_in = os.path.abspath(os.path.join(JSON, name_in))
    mesh = MeshPlus.from_json(json_in)

# ==============================================================================
# Query vector_fields
# ==============================================================================

    vf = mesh.vector_field(vf_name)
    vf_90 = mesh.vector_field(vf_name_90)

# ==============================================================================
# Sort vertices keys in ascending order
# ==============================================================================

    sorted_vkeys = sorted(list(mesh.vertices()))

# ==============================================================================
# Export vertices and faces
# ==============================================================================

    V = np.array([mesh.vertex_coordinates(vkey) for vkey in sorted_vkeys])
    print("V shape: ", V.shape)
    print("V first row: {}".format(V[0,:]))
    print("V last row: {}".format(V[-1,:]))

    F = np.array([mesh.face_vertices(fkey) for fkey in mesh.faces()])
    print("F shape: ", F.shape)
    print("F first row: {}".format(F[0,:]))
    print("F last row: {}".format(F[-1,:]))

    np.savetxt(THERE + "vertices.txt", V, fmt="%1.6f", delimiter=" ", encoding=None)
    np.savetxt(THERE + "faces.txt", F, fmt="%d", delimiter=" ", encoding=None)

# ==============================================================================
# Export vector fields
# ==============================================================================

    PSS = []
    for outname, vector_field in {"ps1.txt": vf, "ps2.txt": vf_90}.items():

        ps = []

        for fkey in mesh.faces():
            ps.append(normalize_vector(vector_field[fkey]))

        PS = np.array(ps)
        print(outname)
        print("Shape: ", PS.shape)
        print("First row: {}".format(PS[0, :]))
        print("Last row: {}".format(PS[-1, :]))

        PSS.append(PS)
        np.savetxt(THERE + outname, PS, fmt="%1.6f", delimiter=" ", encoding=None)

# ==============================================================================
# Export vector fields
# ==============================================================================

    PS1, PS2 = PSS
    print("Dot product first row PS1 - PS2: {}".format(np.dot(PS1[0, :], PS2[0,:].T)))
    print("Ok!")

# ==============================================================================
# Executable
# ==============================================================================

if __name__ == '__main__':
    fire.Fire(export_libigl)
