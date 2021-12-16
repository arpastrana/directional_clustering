from math import pi

# from compas.geometry import normalize_vector
# from compas.geometry import rotate_points
from compas.geometry import Vector
from compas.geometry import Rotation

from directional_clustering.fields import VectorField
from directional_clustering.fields import Field


__all__ = ["NRoSyField"]


class NRoSyField(Field):
    """
    An N-Rotationally Symmetric (RoSy) field.
    An N-RoSy field stores N rotationally-equidistant 3D vector fields.
    The dimensionality of every vector field is 3.
    The dimensionality of the N-RoSy field is N * 3.
    """
    def __init__(self, degree):
        """
        The constructor

        Parameters
        ----------
        `degree` : `int`
            The degree N of the N-RoSy field.
            The number of equidistant vectors in the N-RoSy field.
            Degree must be at least 2.
        """
        assert degree > 1
        super(NRoSyField, self).__init__(dimensionality=degree * 3)

    # --------------------------------------------------------------------------
    # Factory methods
    # --------------------------------------------------------------------------

    @classmethod
    def from_vector_field(cls, vector_field, mesh, degree, unitize=True):
        """
        Create an N-Rotationally Symmetric (RoSy) field from a vector field.

        Parameters
        ----------
        vector_field : `directional_clustering.fields.VectorField`.
            The vector field to take as the basis for rotation.
        mesh : `directional_clustering.mesh.MeshPlus`.
            A mesh whose face normals are used as the basis for rotation.
            The mesh face keys must match the vector field keys.
        `degree` : `int`
            The degree N of the N-RoSy field.
            The number of equidistant vectors in the N-RoSy field.
        unitize : `bool`, optional.
            Indicate whether to unitize the length of input vectors.
            Defaults to `True`.

        Notes
        -----
        To generate the N-RoSy field, each vector in the input vector field
        is CCW rotated a fixed angle `degree` times around the normal of
        its key-matching face in the mesh. The angle of rotation is given
        2 * pi / `degree`.
        """
        nrosy = cls(degree)
        angle = 2 * pi / degree

        for fkey in mesh.faces():

            vector = Vector(*vector_field[fkey])
            if unitize:
                vector.unitize()

            vector_nrosy = [coordinate for coordinate in vector]

            face_normal = mesh.face_normal(fkey)
            face_centroid = mesh.face_centroid(fkey)

            for i in range(degree - 1):
                R = Rotation.from_axis_and_angle(face_normal, angle, face_centroid)
                vector.transform(R)
                vector_nrosy.extend(vector)

            nrosy[fkey] = vector_nrosy

        return nrosy

    # --------------------------------------------------------------------------
    # IO
    # --------------------------------------------------------------------------

    def to_rawfield(filepath):
        """
        Exports a vector field to a .rawfield file.

        Parameters
        ----------
        filepath : `str`
            The filepath where to store the exported (N-Rosy) vector field.

        Notes
        -----
        The vector field is transformed to an N-RoSy field before the export.
        To generate an N-Rotationally Symmetric (RoSy) field, every vector in
        the input field is rotated a fixed angle for `n` times around its
        corresponding face normal. The angle of rotation is 2 * pi / `n`.
        """
        return


if __name__ == "__main__":

    import os
    from directional_clustering import JSON
    from directional_clustering.mesh import MeshPlus

    filename = "four_point_slab"
    vf_name = "m_1"
    degree = 4

    name_in = filename + ".json"
    json_in = os.path.abspath(os.path.join(JSON, name_in))
    mesh = MeshPlus.from_json(json_in)
    vector_field = mesh.vector_field(vf_name)

    nrosy_field = NRoSyField.from_vector_field(vector_field, mesh, degree, True)

    print("Sucess!")
