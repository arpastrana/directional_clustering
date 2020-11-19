from compas.geometry import normalize_vector
from compas.geometry import add_vectors
from compas.geometry import scale_vector
from compas.geometry import length_vector


__all__ = [
    "line_sdl",
    "vector_lines_on_faces",
    "line_tuple_to_dict",
    "polygon_list_to_dict"
]


def line_sdl(start, direction, length, both_sides=True):
    """
    Creates a line from a start point with a given direction and length. It will extend the line in the oposite direction as well unless both_sides is set to False.
    """
    direction = normalize_vector(direction[:])
    a = start
    b = add_vectors(start, scale_vector(direction, +length))
    if both_sides:
        a = add_vectors(start, scale_vector(direction, -length))
    return a, b


def vector_lines_on_faces(mesh, vector_tag, uniform=True, factor=0.02):

    lines = []

    for fkey in mesh.faces():
        vector = mesh.face_attribute(fkey, vector_tag)

        if not vector:
            raise ValueError('Vector {} not defined on face {}'.format(vector_tag, fkey))

        if uniform:
            vec_length = factor
        else:
            vec_length = length_vector(vector) * factor

        pt = mesh.face_centroid(fkey)
        lines.append(line_sdl(pt, vector, vec_length))

    return lines


def line_tuple_to_dict(line):

    a, b = line
    return {'start': a, 'end': b}


def polygon_list_to_dict(polygon):

    return {'points': polygon}


if __name__ == "__main__":
    pass
