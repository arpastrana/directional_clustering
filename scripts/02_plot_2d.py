# os
import os

# argument parsing helper
import fire

# python standard libraries
from itertools import cycle

#JSON file directory
from directional_clustering import JSON

# extended version of Mesh
from directional_clustering.mesh import MeshPlus

# vector field
from directional_clustering.fields import VectorField

#plotters
from directional_clustering.plotters import MeshPlusPlotter


# ==============================================================================
# Plot stuff in 2d
# ==============================================================================

def plot_2d(filename,
            draw_boundary_edges=True,
            draw_faces=True,
            draw_faces_colored=True,
            draw_vector_fields=True):
    """
    Makes a 3d plot of a mesh with a vector field.

    Parameters
    ----------
    filename : `str`
        The name of the JSON file that stores the clustering resultes w.r.t certain
        \n mesh, attribute, alglrithm and number of clusters.

    plot_faces : `bool`
        Plots the faces of the input mesh.
        \nDefaults to True.

    paint_clusters : `bool`
        Color up the faces according to their cluster
        \nDefaults to True.

    plot_mesh_edges : `bool`
        Plots the edges of the input mesh.
        \nDefaults to False.

    plot_vector_fields : `bool`
        Plots the clustered vector field atop of the input mesh.
        \nDefaults to True.

    plot_original_field : `bool`
        Plots the original vector field before clustering atop of the input mesh.
        \nDefaults to False.

    plot_cones : `bool`
        Plots the cones atop of the input mesh.
        \nDefaults to False.
    """
    # load a mesh from a JSON file
    name_in = filename + ".json"
    json_in = os.path.abspath(os.path.join(JSON, name_in))

    mesh = MeshPlus.from_json(json_in)

    # ClusterPlotter is a custom wrapper around a COMPAS MeshPlotter
    plotter = MeshPlusPlotter(mesh, figsize=(16, 9))
    if draw_boundary_edges:
        plotter.draw_edges(keys=list(mesh.edges_on_boundary()))

    # color up the faces of the mesh according to their cluster
    if draw_faces:
        face_colors = None
        if draw_faces_colored:
            face_colors = None
        plotter.draw_faces(facecolor=face_colors, edgewidth=0.01)

    # plot vector fields on mesh as lines
    if draw_vector_fields:
        # supported vector field attributes
        available_vf = mesh.vector_fields()
        print("Avaliable vector fields on the mesh are:\n", available_vf)

        # the name of the vector field to cluster.
        vf_names = set()
        while True:
            vf_name = input("Please select the vector fields to cluster. Type 'ok' to stop adding vector fields: ")
            if vf_name in available_vf:
                vf_names.add(vf_name)
            elif vf_name == "ok":
                break
            else:
                print("This vector field is not available. Please try again.")

        # vector field drawing parameters -- better of being exposed?
        width = 0.5
        length = 0.05
        same_length = True
        colors = cycle([(255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 0, 0)])

        for vf_name in vf_names:
            vf = mesh.vector_field(vf_name)
            color = next(colors)
            plotter.draw_vector_field(vf, color, same_length, length, width)

    # show to screen
    plotter.show()


if __name__ == '__main__':
    fire.Fire(plot_2d)
