# os
import os

# argument parsing helper
import fire

# hello numpy, my old friend
import numpy as np

# plots and beyond
import matplotlib.pyplot as plt

# sprinkle uncertainty into the mix
import random

# python standard libraries
from itertools import cycle

# compas & co
from compas.utilities import geometric_key_xy

#JSON file directory
from directional_clustering import JSON

# extended version of Mesh
from directional_clustering.mesh import MeshPlus

# vector field
from directional_clustering.fields import VectorField

# transformations
from directional_clustering.transformations import align_vector_field

#plotters
from directional_clustering.plotters import MeshPlusPlotter


# ==============================================================================
# Plot stuff in 2d
# ==============================================================================

def plot_2d(filename,
            draw_vector_fields=True,
            draw_vector_lines=False,
            draw_streamlines=False,
            draw_boundary_edges=True,
            draw_faces=True,
            draw_edges=False,
            draw_faces_colored=True,
            density=1.0
            ):
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

    # draw mesh edges
    if draw_edges:
        plotter.draw_edges()

    # color up the faces of the mesh according to their cluster
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
        vf_names = []
        while True:
            vf_name = input("Please select the vector fields to draw. Type 'ok' to stop adding vector fields: ")
            if vf_name in available_vf:
                if vf_name not in vf_names:
                    vf_names.append(vf_name)
            elif vf_name == "ok":
                break
            else:
                print("This vector field is not available. Please try again.")

        # colors for all vector related matters
        colors = cycle([(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 0)])

        if draw_vector_lines:
            # vector field drawing parameters -- better of being exposed?
            width = 0.5
            length = 0.05
            same_length = True

            for vf_name in vf_names:
                vf = mesh.vector_field(vf_name)
                color = next(colors)
                plotter.draw_vector_field(vf, color, same_length, length, width)

        if draw_streamlines:

            # gather all x and y coordinates
            xs = set()
            ys = set()
            gkey_fkey = {}
            gkey_xyz = {}

            for fkey in mesh.faces():
                gkey = geometric_key_xy(mesh.face_centroid(fkey))
                gkey_xyz[gkey] = mesh.face_centroid(fkey)
                gkey_fkey[gkey] = fkey

                x, y = gkey.split(",")
                xs.add(float(x))
                ys.add(float(y))

            # make a grid out of x and y
            X = sorted(list(xs))
            Y = sorted(list(ys))
            XX, YY = np.meshgrid(X, Y)

            # TODO: hack for triangular square-ish meshes
            XX = XX[1::2, 1::2]
            YY = YY[::2, ::2]

            for vf_name in vf_names:

                vf = mesh.vector_field(vf_name)

                alignment_ref = [0, 1, 0]
                align_vector_field(vf, alignment_ref)

                # query vectors from vector field
                U = []
                V = []
                for xx, yy in zip(XX.flatten(), YY.flatten()):
                    gkey = geometric_key_xy([xx, yy, 0.0])

                    try:
                        fkey = gkey_fkey[gkey]
                        # grab only the first two coordinates
                        u, v = vf[fkey][:2]
                    except KeyError:
                        u = np.nan
                        v = np.nan

                    U.append(u)
                    V.append(v)

                U = np.reshape(U, XX.shape)
                V = np.reshape(V, XX.shape)

                # plot streamlines
                plt.streamplot(XX, YY, U, V,
                               color=[i / 255.0 for i in next(colors)],
                               arrowsize=0.0,
                               maxlength=20.0,
                               minlength=0.1,
                               density=density)

    # show to screen
    plotter.show()


if __name__ == '__main__':
    fire.Fire(plot_2d)
