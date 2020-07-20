from compas.geometry import dot_vectors
from compas.geometry import length_vector
from compas.geometry import rotate_points


__all__ = [
    "vector_from_angle",
    "vectors_from_angles",
    "cosine_similarity"
    ]


def vector_from_angle(angle, base_vector):

    rot_pts = rotate_points([base_vector], angle)
    return rot_pts[0]


def vectors_from_angles(angles, base_vector):

    vectors = {}
    
    for fkey, angle in angles.items():
        rot_pts = rotate_points([base_vector], angle)
        vectors[fkey] = vector_from_angle(angle, base_vector)

    return vectors


def cosine_similarity(u, v):

    return dot_vectors(u, v) / (length_vector(u) * length_vector(v))


if __name__ == "__main__":
    pass
