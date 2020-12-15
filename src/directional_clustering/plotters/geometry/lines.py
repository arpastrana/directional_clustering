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
    Creates a line from a start point with a given direction and length. It will also extend the line in the oposite direction unless both_sides is set to `False`.

    Parameters
    ----------
    start : `list` of `float`, shape(n,), n<=3
        A point defined by XYZ coordinates.
        If Z coordinate is not given, then the results are in 2D.
        If Y and Z coordinates are not given, then the results are in 1D.
    direction : `list` of `float`
        XYZ components of the vector to create a line.
    length : `float`
        The length of the line.
    both_sides : `bool`, optional
        Flag to produce a line in one or both directions.
        Deafault is set to `True`.

    Returns
    -------
    a, b : `tuple`
        Returns the start and end points of the line.
    """
    direction = normalize_vector(direction[:])
    a = start
    b = add_vectors(start, scale_vector(direction, +length))
    if both_sides:
        a = add_vectors(start, scale_vector(direction, -length))
    return a, b


def vector_lines_on_faces(mesh, vector_tag, uniform=True, factor=0.02):
    """
    Creates lines in the direction of the vector in each face of the mesh.

    Parameters
    ----------
    mesh : `compas.datastructures.Mesh`
        A COMPAS mesh with vectors as face attributes.
    vector_tag : `string`
        Name of a vector field stored on the faces of a mesh.
    uniform : `bool`, optional
        Constructs lines with the same length if `True`, otherwise length is just scaled by factor.
        Default is set to `True`.
    factor : `float`, optional
        This factor will determine either half the size of the line (created from line_sdl with both_sides set to `True`) or the factor that scales the mesh face vector.
        Default is set to `0.02`.

    Returns
    -------
    lines : `list` of `tuple`
        Returns lines in the direction of the given face vector centered in the centroid of each mesh face.
    """
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
    """
    Returns a dictionary with start and end points of a line.

    Parameters
    ----------
    line : `tuple`
        A tuple with two points.

    Returns
    -------
    line_dict : `dict`
        A dictionary of with the start and end points of a line.
    """
    a, b = line
    return {'start': a, 'end': b}


def polygon_list_to_dict(polygon):
    """
    Returns a dictionary with the list of points of a polygon.

    Parameters
    ----------
    polygon : `list`
        A list with the XYZ coordinates of the vertices of a polygon.

    Returns
    -------
    polygon_dict : `dict`
        A dictionary of with the points of a polygon.
    """
    return {'points': polygon}


if __name__ == "__main__":
    pass
