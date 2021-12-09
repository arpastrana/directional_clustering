import numpy as np

from igl import comb_line_field

from directional_clustering.fields import VectorField


__all__ = ["comb_vector_field"]


def comb_vector_field(vector_field, mesh):
    """
    Combs a vector field defined on a reference triangular mesh.

    Parameters
    ----------
    vector_field : `directional_clustering.fields.VectorField`
        A vector field.
    mesh : `directional_clustering.mesh.MeshPlus`
        The reference triaangular mesh.

    Notes
    -----
    This function uses numpy and libigl to comb a field.
    The mesh must be composed only by triangular faces.
    """
    # mesh information
    F = []
    for fkey in mesh.faces():
        face_indices = mesh.face_vertices(fkey)
        assert len(face_indices) == 3
        F.append(face_indices)

    VE = []
    for vkey in sorted(list(mesh.vertices())):
        VE.append(mesh.vertex_coordinates(vkey))

    # numpify mesh information
    F = np.reshape(np.array(F), (-1, 3))
    VE = np.reshape(np.array(VE), (-1, 3))

    VF = [vector_field[fkey] for fkey in mesh.faces()]
    VF = np.reshape(np.array(VF), (-1, 3))

    combed_vf = comb_line_field(VE, F, VF)
    vf = VectorField()
    for idx, fkey in enumerate(mesh.faces()):
        vf.add_vector(fkey, combed_vf[idx, :].tolist())

    return vf


if __name__ == "__main__":
    pass
