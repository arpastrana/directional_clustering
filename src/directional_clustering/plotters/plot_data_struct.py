from numpy import asarray
from numpy import zeros
from numpy import empty

__all__ = [
    "mesh_to_vertices_xyz",
    "trimesh_face_connect",
    "lines_to_start_end_xyz",
    "lines_xyz_to_tables",
    "coord_start_end_none",
    "lines_start_end_connect",
    "vectors_dict_to_array",
    "face_centroids"
]


def mesh_to_vertices_xyz(mesh):
    """
    Organizes data structure. Splits the vertices of a COMPAS mesh into lists of x, y, and z coordinate.

    Parameters
    ----------
    mesh : `compas.datastructures.Mesh`
        A COMPAS mesh with 3D vertices.

    Returns
    -------
    vertices_x, vertices_y, vertices_z : `tuple`
        Returns lists of x, y, z coordinates of a mesh.
    """
    mesh_vertices, _ = mesh.to_vertices_and_faces()
    vertices_arr = asarray(mesh_vertices)

    # TODO: test if vertices have 3D coordinates

    # get separate arrays for x, y, z coords. of vertices
    vertices_x = vertices_arr[:,0]
    vertices_y = vertices_arr[:,1]
    vertices_z = vertices_arr[:,2]

    return vertices_x, vertices_y, vertices_z


def trimesh_face_connect(mesh):
    """
    Organizes data structure. Splits triangulated COMPAS mesh faces into lists of vertex indices connectivity.

    Parameters
    ----------
    mesh : `compas.datastructures.Mesh`
        A triangulated COMPAS mesh

    Returns
    -------
    face_vertex_i, face_vertex_j, face_vertex_k : `tuple` of `array`
        Returns lists of i, j, k indices of mesh faces.
    """
    _, mesh_faces = mesh.to_vertices_and_faces()
    faces_arr = asarray(mesh_faces)

    # TODO: test if mesh is triangulated

    # get separate arrays for vertex connectivity i, j, k
    face_vertex_i = faces_arr[:,0]
    face_vertex_j = faces_arr[:,1]
    face_vertex_k = faces_arr[:,2]

    return face_vertex_i, face_vertex_j, face_vertex_k


def lines_to_start_end_xyz(lines):
    """
    Arranges start and end points of lines into lists of xyz coordinates.

    Parameters
    ----------
    lines : `list` of tuple`
        Start and end point of a line.

    Returns
    -------
    start_x, start_y, start_z, end_x, end_y, end_z : `tuple`
        Returns lists of coordinates of start and end points of lines.
    """

    num_lines = len(lines)

    start = zeros((num_lines,3))
    end = zeros((num_lines,3))

    for l in range(num_lines):
        start[l], end[l] = lines[l]

    start_x = start[:,0]
    start_y = start[:,1]
    start_z = start[:,2]

    end_x = end[:,0]
    end_y = end[:,1]
    end_z = end[:,2]

    return start_x, start_y, start_z, end_x, end_y, end_z


def lines_xyz_to_tables(start_x, start_y, start_z, end_x, end_y, end_z):
    """
    Arranges lists of xyz coordinates of lines into tables of start and end coordinate.

    Parameters
    ----------
    start_x, start_y, start_z, end_x, end_y, end_z : `tuple`
        Lists of coordinates of start and end points of lines.

    Returns
    -------
    table_x, table_y, table_z : `tuple` of `list`
        2D lists of start and end coordinates of lines.
    """

    num_lines = len(start_x)

    table_x = zeros((num_lines,2)) # start and end point x coord.
    table_y = zeros((num_lines,2)) # start and end point y coord.
    table_z = zeros((num_lines,2)) # start and end point z coord.

    for i in range(num_lines):
        table_x[i] = [start_x[i], end_x[i]]
        table_y[i] = [start_y[i], end_y[i]]
        table_z[i] = [start_z[i], end_z[i]]

    return table_x, table_y, table_z

def coord_start_end_none(nums_1st, nums_2nd, num_lines):
    """
    Helper function to lines_start_end_connect.
    Orders a list with 1st number of 1st list, 2nd number of 2nd list, and None; then 2nd number of 1st list, 2nd number of 2nd list, and None; and so on, until the end of the list.
    Parameters
    ----------
    nums_1st, nums_2nd : `list`
        List of numbers to reorder.

    Returns
    -------
    connect : `list`
        Combined list of 1st number, 2nd number, and `nan` as separator.
    """

    connect = empty(3 * num_lines)
    connect[::3] = nums_1st
    connect[1::3] = nums_2nd
    connect[2::3] = None

    return connect

def lines_start_end_connect(start_x, start_y, start_z, end_x, end_y, end_z):
    """
    Arranges connectivity of lines into a table with respective coordinates to connect and None to separate each line coordinate.

    Parameters
    ----------
    start_x, start_y, start_z, end_x, end_y, end_z : `list`
        Lists of coordinates of start and end points of lines.

    Returns
    -------
    connected_x, connected_y, connected_z : `tuple`
        Lists of start and end coordinates of lines with `nan` as separators.
    """
    num_lines = len(start_x)

    connected_x = coord_start_end_none(start_x, end_x, num_lines)
    connected_y = coord_start_end_none(start_y, end_y, num_lines)
    connected_z = coord_start_end_none(start_z, end_z, num_lines)

    return connected_x, connected_y, connected_z

def vectors_dict_to_array(vectors, num_faces):
    """
    Returns an array of vectors when a dictionary of vectors is passed in.

    Parameters
    ----------
    vectors : `dict`
        A dictionary of vectors.
    num_faces : `int`
        Number of faces in the mesh.

    Returns
    -------
    vectors_array : `ndarray`
        Array of vectors
    """
    if type(vectors) is not dict:
        raise TypeError

    # convert vectors dictionary into a numpy array
    vectors_array = zeros((num_faces, 3))
    for fkey, vec in vectors.items():
        vectors_array[fkey, :] = vec
    return vectors_array

def face_centroids(mesh):
    """
    Arranges the centroids of a mesh into lists of each xyz coordinate.

    Parameters
    ----------
    mesh : `compas.datastructures.Mesh`
        A COMPAS mesh.

    Returns
    -------
    c_x, c_y, c_z : `tuple`
        Returns lists of x, y, z coordinates of the centroids of a mesh.
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
