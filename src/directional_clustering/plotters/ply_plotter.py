from numpy import zeros
from numpy import asarray

# plotter is based on Plotly (https://plotly.com/python/)
import plotly.graph_objects as go
import plotly.figure_factory as ff

from compas.geometry import length_vector

from directional_clustering.plotters import mesh_to_vertices_xyz
from directional_clustering.plotters import lines_to_start_end_xyz
from directional_clustering.plotters import lines_start_end_connect
from directional_clustering.plotters import vectors_dict_to_array
from directional_clustering.plotters import face_centroids
from directional_clustering.plotters import line_sdl
from directional_clustering.plotters import rgb_colors


__all__ = ["PlyPlotter"]


# TODO: see decorator style
class PlyPlotter(go.Figure):
    """
    .. autoclass:: PlyPlotter


    A web plotter for 3D geometry.

    Parameters
    ----------
    args : `list`, optional
        Additional arguments.
    kwargs : `dict`, optional
        Additional keyword arguments.

    Notes
    -----
    This is a custom wrapper around a Plotly.py graph object plotter.
    """
    def __init__(self, *args, **kwargs):
        """
        The constructor.
        """
        super(PlyPlotter, self).__init__(*args, **kwargs)

    def set_title(self, title):
        """
        Sets title of the plot and sets the aspect ratio to the data.

        Parameters
        ----------

        Notes
        -----
        The default aspect ratio is of a cube which may visually distort the mesh.
        """
        self.update_layout(title_text=title,
                           showlegend=False,
                           scene=dict(aspectmode='data'))

    def plot_vector_field_lines(self, mesh, vector_field, color, uniform, scale, width):
        """
        Plots a vector field.

        Parameters
        ----------
        """
        field = vectors_dict_to_array(vector_field, mesh.number_of_faces())
        lines = []

        # get lines
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

        # add lines to plot
        self.add_trace(go.Scatter3d(x=cse_x,
                                    y=cse_y,
                                    z=cse_z,
                                    mode='lines',
                                    line=dict(width=2, color=color),
                                    opacity=1))

    def plot_trimesh(self, mesh, paint_clusters, plot_edges=False, opacity=0.8):
        """
        Plots a triangulated mesh in color.

        Parameters
        ----------

        Notes
        -----
        The colors of the mesh faces are based their the cluster labels.
        """
        # color up the faces of the COMPAS mesh according to their cluster
        # make a dictionary with all labels
        if paint_clusters:
            labels_to_color = {}
            for fkey in mesh.faces():
                labels_to_color[fkey] = mesh.face_attribute(key=fkey, name="cluster")
            # convert labels to rgb colors
            face_colors = rgb_colors(labels_to_color, invert=False)
            face_colors = list(face_colors.values())
        else:
            face_colors = [(255,255,255) for i in range(mesh.number_of_faces())]

        # "v" is shorthand for vertices
        v_x, v_y, v_z = mesh_to_vertices_xyz(mesh)

        _, mesh_faces = mesh.to_vertices_and_faces()

        # we must go for another type of plot if we want to have the option of
        # ploting mesh edges down the line
        figure_mesh = ff.create_trisurf(x=v_x,
                                        y=v_y,
                                        z=v_z,
                                        simplices=asarray(mesh_faces),
                                        color_func=face_colors)

        self.add_trace(figure_mesh.data[0]) # adds mesh faces

        if plot_edges:
            self.add_trace(figure_mesh.data[1]) # adds mesh lines

        self.update_traces(opacity=opacity)

    def plot_vector_field_cones(self, mesh, vector_field):
        """
        Plots a vector field as cones.

        Parameters
        ----------

        Notes
        -----
        Automatic color scale is set to vary with magnitude of the vector.
        """
        num_faces = mesh.number_of_faces()
        field = vectors_dict_to_array(vector_field, num_faces)

        c_x, c_y, c_z = face_centroids(mesh)

        self.add_trace(go.Cone(x=c_x,
                               y=c_y,
                               z=c_z,
                               u=field[:, 0],
                               v=field[:, 1],
                               w=field[:, 2]))


if __name__ == "__main__":
    pass
