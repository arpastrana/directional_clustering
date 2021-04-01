# mathilicious
from math import fabs

# os
import os

# argument parsing helper
import fire

# hello numpy, my old friend
import numpy as np

# plots and beyond
import matplotlib.pyplot as plt

# time is running out
from datetime import datetime

# compas & co
from compas.geometry import angle_vectors

# Library file directories
from directional_clustering import DATA
from directional_clustering import JSON

# extended version of Mesh
from directional_clustering.mesh import MeshPlus

# transformations
from directional_clustering.transformations import principal_stresses_and_angles
from directional_clustering.transformations import transformed_stresses
from directional_clustering.transformations import bending_stresses

# plotters
from directional_clustering.plotters import MeshPlusPlotter

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
            align_field_1_to=None,
            align_field_2_to=None,
            streamlines_density=0.75,
            save_img=False,
            pad_inches=0.0,
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

        if color_faces == "bending_stress":
            HEIGHT = 0.15 * 100  # shell thickness, centimeters
            REF_VECTOR = [1.0, 0.0, 0.0]
            moment_names = ["mx", "my", "mxy"]

            # select a vector field to take as the basis for transformation
            available_vf = mesh.vector_fields()
            print("Avaliable vector fields on the mesh are:\n", available_vf)

            while True:
                vf_name = input("Please select a vector field to take as the basis for transformation: ")
                if vf_name in available_vf:
                    break
                else:
                    print("This vector field is not available. Please try again.")

            vf = mesh.vector_field(vf_name)
            # TODO: taking m_1 as reference. does it always hold?
            vf_ref = mesh.vector_field("m_1")

            # choose the bending moment to display
            print("Avaliable bending moments on the mesh are:\n", moment_names)

            while True:
                b_name = input("Please select the bending moment to display: ")
                if b_name in moment_names:
                    break
                else:
                    print("This bending moment is not available. Please try again.")

            # choose the layer to compute the stresses at
            shell_layers = {"top": 1.0, "bottom": -1.0, "bending": 0.0}
            print("Avaliable bending moments on the mesh are:\n", shell_layers)

            while True:
                l_name = input("Please select the layer to compute the stresses at: ")
                if l_name in shell_layers:
                    z_factor = shell_layers[l_name]
                    break
                else:
                    print("This layer is not available. Please try again.")

            # transform bending moments
            data = np.zeros(mesh.number_of_faces())
            sorted_fkeys = sorted(list(mesh.faces()))

            for fkey in sorted_fkeys:
                # get bending information
                mx, my, mxy = mesh.face_attributes(fkey, names=moment_names)
                # generate principal bending moments
                m1a, m2a = principal_stresses_and_angles(mx, my, mxy)
                m1, angle1 = m1a
                # compute delta between reference vector and principal bending vector
                # TODO: will take m1 as reference. does this always hold?
                vec_ref = vf_ref[fkey]
                delta = angle1 - angle_vectors(vec_ref, REF_VECTOR)

                # add delta to target transformation vector field
                vec = vf[fkey]
                theta = delta + angle_vectors(vec, REF_VECTOR)

                # transform bending moments with theta
                btrans = transformed_stresses(mx, my, mxy, theta)

                # convert bending moments into stress
                if z_factor:
                    bx, by, bxy = [b * 1.0 for b in btrans]
                    z = HEIGHT * z_factor * 0.5
                    btrans = bending_stresses(bx, by, bxy, z, HEIGHT)

                bmap = {k: v for k, v in zip(moment_names, btrans)}

                # store the relevant transformed bending
                data[fkey] = bmap[b_name]

                # convert name for display
                bname_map = {"mx": "m1", "my": "m2", "mxy": "m12"}
                bname_out = bname_map[b_name]

                # create label for color map
                if draw_colorbar:
                    if l_name == "bending":
                        cbar_label = "{}: Bending Moment [kNm/m]".format(bname_out)
                    else:
                        cbar_label = "{}: {}: Stress [kN/cm2]".format(bname_out, l_name)

            cmap = "Spectral"  # Spectral, BrBG, viridis, PiYG
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

    # save image
    if save_img:
        dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        img_name = filename.split("/")[-1] + "_" + dt + ".png"
        img_path = os.path.abspath(os.path.join(DATA, "images", img_name))
        plt.tight_layout()
        plotter.save(img_path, bbox_inches='tight', pad_inches=pad_inches)
        print("Saved image to : {}".format(img_path))

    # show to screen
    if show_img:
        plotter.show()



if __name__ == '__main__':
    fire.Fire(plot_2d)
