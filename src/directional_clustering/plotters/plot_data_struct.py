from numpy import asarray
from numpy import zeros
from numpy import empty

__all__ = [
    "mesh_to_vertices_xyz",
    "trimesh_to_face_connected",
    "lines_to_start_end_xyz",
    "lines_xyz_to_tables",
    "lines_start_end_connected"
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

def trimesh_to_face_connected(mesh):
    """
    Organizes data structure. Splits triangulated COMPAS mesh faces into lists of vertex indices connectivity.

    Parameters
    ----------
    mesh : `compas.datastructures.Mesh`
        A triangulated COMPAS mesh

    Returns
    -------
    face_vertex_i, face_vertex_j, face_vertex_k : `tuple`
        Returns lists of i, j, k indices of mesh faces.
    """
    _, mesh_faces = mesh.to_vertices_and_faces()
    faces_arr = asarray(mesh_faces)

    # TODO: test if mesh is not triangulated

    # get separate arrays for vertex connectivity i, j, k
    face_vertex_i = faces_arr[:,0]
    face_vertex_j = faces_arr[:,1]
    face_vertex_k = faces_arr[:,2]

    return face_vertex_i, face_vertex_j, face_vertex_k

def lines_to_start_end_xyz(lines):
    """

    Parameters
    ----------

    Returns
    -------

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

    Parameters
    ----------

    Returns
    -------

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

def lines_start_end_connected(start_x, start_y, start_z, end_x, end_y, end_z):
    """

    Parameters
    ----------

    Returns
    -------

    """
    num_lines = len(start_x)

    connected_x = []
    connected_y = []
    connected_z = []

    connected_x = empty(3 * num_lines)
    connected_x[::3] = start_x
    connected_x[1::3] = end_x
    connected_x[2::3] = None

    connected_y = empty(3 * num_lines)
    connected_y[::3] = start_y
    connected_y[1::3] = end_y
    connected_y[2::3] = None

    connected_z = empty(3 * num_lines)
    connected_z[::3] = start_z
    connected_z[1::3] = end_z
    connected_z[2::3] = None

    return connected_x, connected_y, connected_z


if __name__ == "__main__":
    pass
