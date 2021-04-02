from math import sqrt
from math import pi
from math import atan
from math import cos
from math import sin
from math import fabs

from compas.geometry import angle_vectors
from compas.geometry import scale_vector
from compas.geometry import normalize_vector

from directional_clustering.fields import VectorField


__all__ = ["principal_stresses",
           "principal_stresses_and_angles",
           "principal_angles",
           "transformed_stresses",
           "bending_stresses",
           "transformed_stress_vector_fields"]


def principal_stresses(fx, fy, fxy):
    """
    Calculates the magnitude of the principal stresses.
    """
    pf_a = (fx + fy) / 2
    pf_b = sqrt(((fx - fy) / 2) ** 2 + fxy ** 2)

    return pf_a + pf_b, pf_a - pf_b


def principal_angles(fx, fy, fxy):
    """
    Calculates the angles, relative to an XY plane, of the principal stresses.

    Note
    ----
    The angles are unordedered.
    """
    a = 2 * fxy / (fx - fy)
    b = atan(a) / 2

    return b, b + pi / 2


def principal_stresses_and_angles(fx, fy, fxy, tol=0.001):
    """
    Computes the angles corresponding to each principal stress direction.

    Notes
    -----
    The angles are ordered such that the first angle matches the first stress.
    """
    pf1, pf2 = principal_stresses(fx, fy, fxy)
    angle1, angle2 = principal_angles(fx, fy, fxy)
    tf1, _, _ = transformed_stresses(fx, fy, fxy, angle1)

    if fabs(fabs(tf1) - fabs(pf1)) < tol:
        return (pf1, angle1), (pf2, angle2)

    return (pf1, angle2), (pf2, angle1)


def transformed_stresses(fx, fy, fxy, theta):
    """
    Transforms the stresses given an angle relative to an XY plane.
    """

    tfx = (fx + fy) / 2 + (fx - fy) * cos(2 * theta) / 2 + fxy * sin(2 * theta)
    tfy = fx + fy - tfx

    # TODO: double check sign flips in the torsion component
    tfxy = -(fx - fy) * sin(2 * theta) / 2 + fxy * cos(2 * theta)

    return tfx, tfy, tfxy


def bending_stresses(mx, my, mxy, z, h):
    """
    Computes the bending stresses at given shell thickness and fiber height.
    """
    def stress(m):
        return m * z / ((h ** 3) / 12)

    return stress(-mx), stress(-my), stress(mxy)


def transformed_stress_vector_fields(mesh, vector_fields, stress_type, ref_vector):
    """
    Rescales a vector field based on a plane stress transformation.
    """
    vf1, vf2  = vector_fields

    # TODO: mapping is not robust! depends on naming convention
    stress_components = {"bending": {"names": ["mx", "my", "mxy"], "ps": "m_1"},
                         "axial":  {"names": ["nx", "ny", "nxy"], "ps": "n_1"}}

    stress_names = stress_components[stress_type]["names"]
    vf_ps = mesh.vector_field(stress_components[stress_type]["ps"])

    vf1_transf = VectorField()
    vf2_transf = VectorField()

    for fkey in mesh.faces():
        # query stress components
        sx, sy, sxy = mesh.face_attributes(fkey, names=stress_names)

        # generate principal stresses and angles
        s1a, s1 = principal_stresses_and_angles(sx, sy, sxy)
        s1, angle1 = s1a

        vector_ps = vf_ps[fkey]
        vector1 = vf1[fkey]
        vector2 = vf2[fkey]

        # compute delta between reference vector and principal bending vector
        # TODO: will take m1 as reference. does this always hold?

        delta = angle1 - angle_vectors(vector_ps, ref_vector)
        # add delta to angles of the vector field to transform
        theta = delta + angle_vectors(vector1, ref_vector)

        # transform stresses - this becomes the scale of the vectors
        s1, s2, _ = transformed_stresses(sx, sy, sxy, theta)

        vf1_transf.add_vector(fkey, scale_vector(normalize_vector(vector1), s1))
        vf2_transf.add_vector(fkey, scale_vector(normalize_vector(vector2), s2))

    return vf1_transf, vf2_transf


if __name__ == "__main__":
    import numpy as np

    from math import radians

    sigmax = -80
    sigmay = 50
    sigmaxy = -25
    theta = -30  # -30 degrees

    for i in (45, 135, 225, 315):
        sigmai = transformed_stresses(sigmax, sigmay, sigmaxy, radians(i))
        print(i, sigmai)

    sigmat = transformed_stresses(sigmax, sigmay, sigmaxy, radians(0.0))
    assert np.allclose(np.array(sigmat), np.array([sigmax, sigmay, sigmaxy]))

    sigmat = transformed_stresses(sigmax, sigmay, sigmaxy, radians(theta))
    assert np.allclose(np.array(sigmat), np.array([-25.9, -4.15, -68.8]), rtol=0.01)

    sigmap = principal_stresses(sigmax, sigmay, sigmaxy)
    assert np.allclose(np.array(sigmap), np.array([54.6, -84.6]), rtol=0.01)

    ps1, ps2 = principal_stresses_and_angles(sigmax, sigmay, sigmaxy)
    sigma1, angle1 = ps1
    sigma2, angle2 = ps2
    assert np.allclose(np.array([sigma1, angle1]), np.array([54.6, radians(100.5)]), rtol=0.01)
    assert np.allclose(np.array([sigma2, angle2]), np.array([-84.6, radians(10.5)]), rtol=0.01)
