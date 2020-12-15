from numpy import asarray
from numpy import zeros
from numpy import empty

from directional_clustering.fields import VectorField


__all__ = ["mesh_to_vertices_xyz",
           "trimesh_face_connect",
           "lines_to_start_end_xyz",
           "lines_xyz_to_tables",
           "coord_start_end_none",
           "lines_start_end_connect",
           "vectors_dict_to_array",
           "face_centroids"]


def mesh_to_vertices_xyz(mesh):
    """
    Streams the vertices of a mesh into lists of x, y, and z coordinates.

    Parameters
    ----------
    mesh : `compas.datastructures.Mesh`
        A COMPAS mesh with 3D vertices.

    Returns
    -------
    vertices_x : `np.array`, (n,)
        The x coordinates of the vertices of a mesh. `n` is the number of
        vertices in the mesh.
    vertices_y : `np.array`, (n,)
        The y coordinates of the vertices of a mesh.
    vertices_z : `np.array`, (n,)
        The z coordinates of the vertices of a mesh.

    Notes
    ------
    This is a helper to organize a data structure.
    """
    mesh_vertices, _ = mesh.to_vertices_and_faces()
    vertices_arr = asarray(mesh_vertices)

    # TODO: test if vertices have 3D coordinates

    # get separate arrays for x, y, z coords. of vertices
    vertices_x = vertices_arr[:, 0]
    vertices_y = vertices_arr[:, 1]
    vertices_z = vertices_arr[:, 2]

    return vertices_x, vertices_y, vertices_z


def trimesh_face_connect(mesh):
    """
    Streams the vertex connectivity of a triangulated mesh into lists of vertices.

    Parameters
    ----------
    mesh : `directional_clustering.mesh.MeshPlus`
        A triangulated COMPAS mesh

    Returns
    -------
    face_vertex_i : `np.array`, (n,)
        The key of the first vertex of the mesh faces.
        Where `n` is the number of faces in the mesh.
    face_vertex_j : `np.array`, (n,)
        The key of the second vertex of the mesh faces.
    face_vertex_k : `np.array`, (n,)
        The key of the third vertex of the mesh faces.

    Notes
    ------
    This is a helper to organize a data structure.
    """
    _, mesh_faces = mesh.to_vertices_and_faces()
    faces_arr = asarray(mesh_faces)

    # TODO: test if mesh is triangulated

    # get separate arrays for vertex connectivity i, j, k
    face_vertex_i = faces_arr[:, 0]
    face_vertex_j = faces_arr[:, 1]
    face_vertex_k = faces_arr[:, 2]

    return face_vertex_i, face_vertex_j, face_vertex_k


def lines_to_start_end_xyz(lines):
    """
    Arranges the start and end points of lines into lists of xyz coordinates.

    Parameters
    ----------
    lines : `list` of `tuple`
        A list with the start and end points of a group of lines.

    Returns
    -------
    start_x : `np.array`, (n,)
        The x coordinates of the start point of a line.
        Where `n` is the number of lines.
    start_y : `np.array`, (n,)
        The y coordinates of the start point of a line.
    start_z : `np.array`, (n,)
        The z coordinates of the start point of a line.
    end_x : `np.array`, (n,)
        The x coordinates of the end point of a line.
    end_y : `np.array`, (n,)
        The y coordinates of the end point of a line.
    end_z : `np.array`, (n,)
        The z coordinates of the end point of a line.
    """
    num_lines = len(lines)

    start = zeros((num_lines, 3))
    end = zeros((num_lines, 3))

    for l in range(num_lines):
        start[l], end[l] = lines[l]

    start_x = start[:, 0]
    start_y = start[:, 1]
    start_z = start[:, 2]

    end_x = end[:, 0]
    end_y = end[:, 1]
    end_z = end[:, 2]

    return start_x, start_y, start_z, end_x, end_y, end_z


def lines_xyz_to_tables(start_x, start_y, start_z, end_x, end_y, end_z):
    """
    Arrange the xyz coordinates a group of lines into start and end tables.

    Parameters
    ----------
    start_x : `np.array`, (n,)
        The x coordinates of the start point of a line.
        `n` is the length of the list.
    start_y : `np.array`, (n,)
        The y coordinates of the start point of a line.
    start_z : `np.array`, (n,)
        The z coordinates of the start point of a line.
    end_x : `np.array`, (n,)
        The x coordinates of the end point of a line.
    end_y : `np.array`, (n,)
        The y coordinates of the end point of a line.
    end_z : `np.array`, (n,)
        The z coordinates of the end point of a line.

    Returns
    -------
    table_x : `np.array`, (n, 2)
        The x coordinates of the start and end point of a line.
        `n` is the number of lines.
    table_y : `np.array`, (n, 2)
        The y coordinates of the start and end point of a line.
    table_z : `np.array`, (n, 2)
        The z coordinates of the start and end point of a line.
    """
    num_lines = len(start_x)

    table_x = zeros((num_lines, 2)) # start and end point x coord.
    table_y = zeros((num_lines, 2)) # start and end point y coord.
    table_z = zeros((num_lines, 2)) # start and end point z coord.

    for i in range(num_lines):
        table_x[i] = [start_x[i], end_x[i]]
        table_y[i] = [start_y[i], end_y[i]]
        table_z[i] = [start_z[i], end_z[i]]

    return table_x, table_y, table_z


def coord_start_end_none(nums_1st, nums_2nd, num_lines):
    """
    Merges the entries of two lists into a single stream by inserting NaNs as separator.

    Parameters
    ----------
    nums_1st, nums_2nd : `list`
        List of numbers to reorder.
        Both lists should be of length `num_lines`.
    num_lines : `int`
        The length of the lists to reorder.

    Returns
    -------
    connect : `np.array`, (3, n)
        Combined list of 1st number, 2nd number, and `nan` as separator.
        `n` is `num_lines`.

    Notes
    ------
    Helper function for lines_start_end_connect.

    Orders a list with 1st number of 1st list, 2nd number of 2nd list, and None.
    Then, 2nd number of 1st list, 2nd number of 2nd list, and None; and so on.
    This is repeated until the end of the list is hit.
    """
    connect = empty(3 * num_lines)
    connect[::3] = nums_1st
    connect[1::3] = nums_2nd
    connect[2::3] = None

    return connect


def lines_start_end_connect(start_x, start_y, start_z, end_x, end_y, end_z):
    """
    Creates a connectivity table from lines' endpoints.

    Parameters
    ----------
    start_x, start_y, start_z, end_x, end_y, end_z : `list`
        Lists of coordinates of start and end points of lines.

    Returns
    -------
    connected_x : `np.array` (3 * n,)
        Start and end x coordinates of lines with `nan` as separators.
        `n` is the number of lines.
    connected_y : `np.array` (3 * n,)
        Start and end y coordinates of lines with `nan` as separators.
    connected_z : `np.array` (3 * n,)
         Start and end z coordinates of lines with `nan` as separators.
    """
    num_lines = len(start_x)

    connected_x = coord_start_end_none(start_x, end_x, num_lines)
    connected_y = coord_start_end_none(start_y, end_y, num_lines)
    connected_z = coord_start_end_none(start_z, end_z, num_lines)

    return connected_x, connected_y, connected_z


def vectors_dict_to_array(vector_field, num_faces):
    """
    Converts a vector field into an array.

    Parameters
    ----------
    vector_field : `directional_clustering.fields.VectorField()`
        A vector field.
    num_faces : `int`
        The number of faces in a mesh.

    Returns
    -------
    vectors_array : `np.array` (n, )
        An array of vectors.
    """
    if type(vector_field) is not type(VectorField()):
        raise TypeError

    # convert vectors dictionary into a numpy array
    vectors_array = zeros((num_faces, 3))
    for fkey, vec in vector_field.items():
        vectors_array[fkey, :] = vec

    return vectors_array


def face_centroids(mesh):
    """
    Splits the xyz coordinates of the face centroids of a mesh into separate
    coordinate streams.

    Parameters
    ----------
    mesh : `compas.datastructures.Mesh`
        A mesh.

    Returns
    -------
    c_x : `np.array` (n,)
        The x coordinates of the face centroids of a mesh.
        `n` is the number of faces in the mesh.
    c_y : `np.array` (n,)
        The y coordinates of the face centroids of a mesh.
    c_z : `np.array` (n,)
        The z coordinates of the face centroids of a mesh.
    """
    num_faces = mesh.number_of_faces()

    # "c" is shorthand for centroid
    c_x = zeros(num_faces)
    c_y = zeros(num_faces)
    c_z = zeros(num_faces)

    for fkey in mesh.faces():
        # "c" is shorthand for centroid
        c_x[fkey], c_y[fkey], c_z[fkey] = mesh.face_centroid(fkey)

    return c_x, c_y, c_z


if __name__ == "__main__":
    pass
