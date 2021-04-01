from math import sqrt
from math import pi
from math import atan
from math import cos
from math import sin
from math import fabs
from math import acos

from compas.geometry import length_vector
from compas.geometry import dot_vectors


__all__ = ["principal_stresses",
           "principal_stresses_and_angles",
           "principal_angles",
           "transformed_stresses",
           "bending_stresses"
           ]


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
    tfxy = -(fx - fy) * sin(2 * theta) / 2 + fxy * cos(2 * theta)

    return tfx, tfy, tfxy


def bending_stresses(mx, my, mxy, z, h):
    """
    Computes the bending stresses at given shell thickness and fiber height.
    """
    def stress(m):
        return m * z / ((h ** 3) / 12)

    return stress(-mx), stress(-my), stress(mxy)
