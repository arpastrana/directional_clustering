from math import degrees
from math import fabs

from compas.geometry import scale_vector
from compas.geometry import cross_vectors

from directional_clustering.geometry import vector_from_angle


__all__ = [
    "faces_angles",
    "faces_labels",
    "faces_clustered_field"
    ]


def faces_angles(mesh, vector_tag, ref_vector, func, twoway=False):

    def twoway_angles(u, v, func, deg):
        if not deg:
            return [func(scale_vector(u, sign), v) for sign in [1, -1]]
        return [degrees(func(scale_vector(u, sign), v)) for sign in [1, -1]]


    angles = {}
    for fkey in mesh.faces():
        vector = mesh.face_attribute(fkey, vector_tag)
        u, v = vector, ref_vector
        
        if twoway:
            twangles = twoway_angles(u, v, func=func, deg=False)
            angles[fkey] = min(twangles)
        else:
            angles[fkey] = func(u, v)

    print('min angle', min(angles.values()))
    print('max angle', max(angles.values()))

    return angles


def faces_clustered_field(mesh, cluster_labels, base_tag, target_tag, perp, func):

    x = [1.0, 0.0, 0.0]
    for fkey, angle in cluster_labels.items():

        base_vec = mesh.face_attribute(fkey, name=base_tag)
        delta = angle - func(base_vec, x)  # or +? 

        vec = vector_from_angle(delta, base_vec)
        test_angle = func(vec, x)

        if fabs(test_angle - angle) > 0.001:
            vec = vector_from_angle(-delta, base_vec)

        if perp:
            vec = cross_vectors(vec, [0, 0, 1])

        mesh.face_attribute(fkey, name=target_tag, value=vec)


def faces_labels(mesh, labels, centers):

    centers[centers < 0.1] = 0  # to avoid float rounding precision issues
    
    clustered_data = centers[labels].tolist()

    face_labels = {}
    for idx, fkey in enumerate(mesh.faces()):
        c = clustered_data[idx][0]
        face_labels[fkey] = c

    return face_labels


if __name__ == "__main__":
    pass
