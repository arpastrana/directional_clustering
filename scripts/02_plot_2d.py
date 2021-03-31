# os
import os

# argument parsing helper
import fire

# hello numpy, my old friend
import numpy as np

# plots and beyond
import matplotlib.pyplot as plt

# python standard libraries
from itertools import cycle
from functools import partial

# time is running out
from datetime import datetime

# compas & co
from compas.geometry import angle_vectors
from compas.geometry import KDTree
from compas.utilities import geometric_key_xy

# Library file directories
from directional_clustering import DATA
from directional_clustering import JSON

# extended version of Mesh
from directional_clustering.mesh import MeshPlus

# clustering
from directional_clustering.clustering import distance_cosine
from directional_clustering.clustering import distance_cosine_abs

# transformations
from directional_clustering.transformations import align_vector_field
from directional_clustering.transformations import comb_vector_field

# plotters
from directional_clustering.plotters import MeshPlusPlotter
from directional_clustering.plotters import rgb_colors


# ==============================================================================
# Plot a lot of information in 2d
# ==============================================================================

def plot_2d(filename,
            draw_vector_fields=False,
            draw_streamlines=False,
            draw_boundary_edges=True,
            draw_faces=False,
            draw_faces_centroids=False,
            color_faces=None,
            draw_colorbar=False,
            draw_edges=False,
            comb_fields=False,
            align_fields_to=None,
            streamlines_density=0.75,
            save_img=False,
            show_img=True
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
    plotter = MeshPlusPlotter(mesh, figsize=(16, 9), dpi=300)
    if draw_boundary_edges:
        plotter.draw_edges(keys=list(mesh.edges_on_boundary()))

    # draw mesh edges
    if draw_edges:
        plotter.draw_edges()

    # color up the faces of the mesh according to their cluster
    if draw_faces or draw_faces_centroids:
        cmap = None
        data = np.zeros(mesh.number_of_faces())
        sorted_fkeys = sorted(list(mesh.faces()))

        if color_faces == "clusters":
            # cmap = "Paired"
            labels = mesh.cluster_labels("cluster")
            for fkey, label in labels.items():
                data[fkey] = label

            # plot stuff
            cbar_label = "Directional Clusters"
            n_clusters = int(input("Please input the number of clusters here: "))
            cmap = plt.cm.get_cmap('rainbow', n_clusters)  # plasma or rainbow

            ticks = np.linspace(0, n_clusters - 1, n_clusters + 1) + 0.5 * (n_clusters - 1)/n_clusters
            ticks = ticks[:-1]
            ticks_labels = list(range(1, n_clusters + 1))
            extend = "neither"

        elif color_faces == "angles" or color_faces == "cosine_distance":

            color_label_map = {"angles": {"cmap": "Blues",
                                          "cbar_label": "Angular Difference [Deg]",
                                          "func": partial(angle_vectors, deg=True)},
                               "cosine_distance":  {"cmap": "RdPu",
                                                    "cbar_label": "Cosine Distance [0 - 1]",
                                                    "func": distance_cosine_abs}}

            available_vf = mesh.vector_fields()
            print("Avaliable vector fields on the mesh are:\n", available_vf)

            vf_names = []
            while len(vf_names) < 2:
                vf_name = input("Please select the two vector fields to compare: ")
                if vf_name in available_vf:
                    if vf_name not in vf_names:
                        vf_names.append(vf_name)
                else:
                    print("This vector field is not available. Please try again.")

            vf_name_a, vf_name_b = vf_names
            vf_a = mesh.vector_field(vf_name_a)
            vf_b = mesh.vector_field(vf_name_b)

            clm = color_label_map[color_faces]
            f = clm["func"]
            cmap = clm["cmap"]
            cbar_label = clm["cbar_label"]

            for fkey in mesh.faces():

                data[fkey] = f(vf_a[fkey], vf_b[fkey])
                # data[fkey] = distance_cosine_abs(vf_a[fkey], vf_b[fkey])
                # data[fkey] = angle_vectors(vf_a[fkey], vf_b[fkey], deg=True)

            ticks = np.linspace(data.min(), data.max(), 7)
            ticks_labels = [np.round(x, 2) for x in ticks]
            extend = "both"

        if draw_faces:

            collection = plotter.draw_faces(keys=sorted_fkeys)

        elif draw_faces_centroids:

            points = []
            for fkey in sorted_fkeys:
                point = {}
                point["pos"] = mesh.face_centroid(fkey)
                point["radius"] = 0.03
                point["edgewidth"] = 0.10
                points.append(point)

            collection = plotter.draw_points(points)

        collection.set(array=data, cmap=cmap)
        collection.set_linewidth(lw=0.0)

        if draw_colorbar:
            colorbar = plt.colorbar(collection,
                                    shrink=0.9,
                                    pad=0.01,
                                    extend=extend,
                                    extendfrac=0.05,
                                    ax=plotter.axes,
                                    aspect=30,
                                    orientation="vertical")

            colorbar.set_ticks(ticks)
            colorbar.ax.set_yticklabels(ticks_labels)
            colorbar.set_label(cbar_label, fontsize="xx-large")

    # plot vector fields on mesh as lines
    if draw_vector_fields or draw_streamlines:
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

        if draw_vector_fields:
            # colors = cycle([(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 0)])
            colors = cycle([(235, 45, 125), (0, 165, 0), (0, 0, 255), (0, 0, 0)])
            # colors for all vector related matters

            # vector field drawing parameters -- better of being exposed?
            width = 0.5
            length = 0.05
            same_length = True

            for vf_name in vf_names:
                vf = mesh.vector_field(vf_name)

                if comb_fields:
                    vf = comb_vector_field(vf, mesh)

                color = next(colors)
                plotter.draw_vector_field(vf, color, same_length, length, width)

        if draw_streamlines:
            colors = cycle([(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 0)])

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

            # do for every vector field
            for vf_name in vf_names:

                vf = mesh.vector_field(vf_name)

                # comb the line field prior to streamlines tracing
                if comb_fields:
                    vf = comb_vector_field(vf, mesh)

                if align_fields_to:
                    alignment_ref = align_fields_to
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
                               arrowsize=0.2,
                               maxlength=20.0,
                               minlength=0.1,
                               density=streamlines_density)

    # save image
    if save_img:
        dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        img_name = filename.split("/")[-1] + "_" + dt + ".png"
        img_path = os.path.abspath(os.path.join(DATA, "images", img_name))
        plt.tight_layout()
        plotter.save(img_path, bbox_inches='tight', pad_inches=0)
        print("Saved image to : {}".format(img_path))

    # show to screen
    if show_img:
        plotter.show()


if __name__ == '__main__':
    fire.Fire(plot_2d)
