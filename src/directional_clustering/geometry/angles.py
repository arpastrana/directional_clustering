from numpy import array
from numpy import mean

from math import acos
from math import atan2
from math import pi

from compas.geometry import angle_vectors
from compas.geometry import dot_vectors
from compas.geometry import length_vector
from compas.geometry import angle_vectors_signed


__all__ = [
    "cos_angle",
    "raw_cos_angle",
    "signed_angle",
    "clockwise",
    "smoothed_angles"
]


def cos_angle(u, v):
    return angle_vectors(u, v)


def raw_cos_angle(u, v):
    return acos(dot_vectors(u, v) / (length_vector(u) * length_vector(v)))


def signed_angle(u, v, normal=[0.0, 0.0, 1.0]):
    return angle_vectors_signed(u, v, normal)


def clockwise(u, v):
    beta = atan2(u[1], u[0])
    return (beta + pi) % (pi)


def smoothed_angles(mesh, angles, smooth_iters):

    smoothed = {}
    for _ in range(smooth_iters):
        averaged_angles = {}

        for fkey in mesh.faces():
            nbrs = mesh.face_neighbors(fkey)
            nbrs.append(fkey)
            local_angles = [angles[key] for key in nbrs]
            
            averaged_angles[fkey] = mean(array(local_angles))

        for fkey in mesh.faces():
            smoothed[fkey] = averaged_angles[fkey]

    return smoothed


if __name__ == "__main__":
    pass
