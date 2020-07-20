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
    "smoothed_angles",
    "laplacian_smoothed"
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


def laplacian_smoothed(mesh, data, iters, damping=0.5):
    
    data = {k: v for k, v in data.items()}

    for i in range(iters):

        iter_smoothed = {}

        for fkey in mesh.faces():
            f_data = data[fkey]
            nbr_data = array([data[key] for key in mesh.face_neighbors(fkey)])
            nbr_data = mean(nbr_data, axis=0)
            iter_smoothed[fkey] = f_data + (1 - damping) * (nbr_data - f_data)

        for fkey in mesh.faces():
            data[fkey] = iter_smoothed[fkey]

    return data


if __name__ == "__main__":
    pass
