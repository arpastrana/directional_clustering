from directional_clustering.plotters import mesh_to_vertices_xyz
from directional_clustering.plotters import trimesh_face_connect
from directional_clustering.plotters import lines_to_start_end_xyz
from directional_clustering.plotters import lines_xyz_to_tables
from directional_clustering.plotters import lines_start_end_connect

from directional_clustering.plotters import line_tuple_to_dict
from directional_clustering.plotters import vector_lines_on_faces
from directional_clustering.plotters import line_sdl

from compas.geometry import length_vector

import plotly.graph_objects as go
import plotly.figure_factory as ff

from numpy import asarray

__all__ = [
    "ply_layout",
    "ply_draw_vector_field_array",
    "ply_draw_trimesh"
]

# TODO: turn the functions in this file into a class (decorator)

def ply_layout(figure, title):

    figure.update_layout(
        title_text = title,
        showlegend = False,
        scene = dict(aspectmode = 'data'))

    return figure


def ply_draw_vector_field_array(figure, mesh, field, color, uniform, scale, width=0.5):
    lines = []

    rows, _ = field.shape
    for fkey in range(rows):
        vector = field[fkey]
        vector = vector.tolist()

        if uniform:
            vec_length = scale
        else:
            vec_length = length_vector(vector) * scale

        pt = mesh.face_centroid(fkey)
        lines.append(line_sdl(pt, vector, vec_length))

    # "s" is shorthand for start, "e" is shorthand for end
    s_x, s_y, s_z, e_x, e_y, e_z = lines_to_start_end_xyz(lines)

    # "cse" is shorthand for connected start and end
    cse_x, cse_y, cse_z = lines_start_end_connect(s_x, s_y, s_z, e_x, e_y, e_z)

    figure.add_trace(
        go.Scatter3d(
            x = cse_x, y = cse_y, z = cse_z,
            mode = 'lines',
            line = dict(width = 2,color = 'black'),
            opacity = 1
        ))

    return figure


def ply_draw_trimesh(figure, mesh, face_colors, draw_edges=False, opacity=0.8):

    # "v" is shorthand for vertices
    v_x, v_y, v_z = mesh_to_vertices_xyz(mesh)

    _, mesh_faces = mesh.to_vertices_and_faces()

    figure_mesh = ff.create_trisurf(
        x = v_x, y = v_y, z = v_z,
        simplices = asarray(mesh_faces),
        color_func = list(face_colors.values())
        )

    figure.add_trace(figure_mesh.data[0]) # adds mesh faces

    if draw_edges:
        figure.add_trace(figure_mesh.data[1]) # adds mesh lines

    figure.update_traces(opacity=opacity)

    return figure


def ply_draw_vector_field_cones(figure, mesh, field):

    for fkey in mesh.faces():
        # "c" is shorthand for centroid
        c_x, c_y, c_z = mesh.face_centroid(fkey)

    figure.add_trace(
        data = go.Cone(
            x=c_x, y=c_y, z=c_z,
            u=field[:,0], v=field[:,1], w=field[:,2]
            ))

    return figure



