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

# shady function tools
from functools import partial

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

# analysis
from directional_clustering.transformations import strain_energy_bending
from directional_clustering.transformations import virtual_work_bending
from directional_clustering.transformations import volume_reinforcement_bending
from directional_clustering.transformations import volume_reinforcement_bending_dir

# plotters
from directional_clustering.plotters import MeshPlusPlotter

# ==============================================================================
# Plot a lot of information in 2d
# ==============================================================================

def plot_2d(filename,
            plate_height,
            e_modulus,
            poisson=0,
            basis_direction="X",
            draw_boundary_edges=True,
            draw_faces=True,
            draw_faces_centroids=False,
            draw_colorbar=True,
            draw_edges=False,
            save_img=True,
            show_img=False
            ):
    """
    Calculates the bending strain energy in a plate.

    Parameters
    ----------
    filename : `str`
        The name of the JSON file that stores the clustering resultes w.r.t certain
        \n mesh, attribute, alglrithm and number of clusters.
    """

    moment_names = ["mx", "my", "mxy"]
    basis_vectors = {"X": [1, 0, 0], "Y": [0, 1, 0]}
    basis_vector = basis_vectors[basis_direction]

    # ==========================================================================
    # Load Mesh
    # ==========================================================================

    # load a mesh from a JSON file
    name_in = filename + ".json"
    json_in = os.path.abspath(os.path.join(JSON, name_in))

    mesh = MeshPlus.from_json(json_in)

    # ==========================================================================
    # Select vector fields
    # ==========================================================================

    # select a vector field along which the strain energy will be calculated
    available_vf = mesh.vector_fields()
    print("Avaliable vector fields on the mesh are:\n", available_vf)

    while True:
        vf_name = input("Select a vector field to calculate the metric along: ")
        if vf_name in available_vf:
            break
        print("This vector field is not available. Please try again.")

    vf = mesh.vector_field(vf_name)

    # select a vector field to take as the basis reference
    while True:
        msg = "Now choose a field as the PS directional basis. Often m_1: "
        vf_basis_name = input(msg)
        if vf_basis_name in available_vf:
            break
        print("This vector field is not available. Please try again.")

    # TODO: taking m_1 as reference. does it always hold?
    vf_ps = mesh.vector_field(vf_basis_name)

    # ==========================================================================
    # Choose structural metric to compute
    # ==========================================================================

    structural_metrics = {"energy": {"cbar_legend": "Strain Energy [kNm]"},
                          "work": {"cbar_label": "Work [kNm]"},
                          "volume": {"func": volume_reinforcement_bending,
                                     "cbar_label": "Volume [KNm]"},
                          "volume1": {"func": volume_reinforcement_bending_dir,
                                      "cbar_label": "Volume 1 [KNm]"},
                          "volume2": {"func": volume_reinforcement_bending_dir,
                                      "cbar_label": "Volume 2 [KNm]"}}

    energy_func = partial(strain_energy_bending, height=plate_height, e_modulus=e_modulus, poisson=poisson)
    structural_metrics["energy"]["func"] = energy_func

    work_func = partial(virtual_work_bending, height=plate_height, e_modulus=e_modulus, poisson=poisson)
    structural_metrics["work"]["func"] = work_func

    # Choose metric
    while True:
        msg = "What metric should I compute, energy, work, volume, volume1, volume2? "
        metric_name = input(msg)
        if metric_name in structural_metrics:
            break
        print("This metric is not available. Please try again.")

    structure_func = structural_metrics[metric_name]["func"]
    cbar_label = structural_metrics[metric_name]["cbar_label"]

    # ==========================================================================
    # Calculate bending strain energy
    # ==========================================================================

    # transform bending moments
    data = np.zeros(mesh.number_of_faces())
    sorted_fkeys = sorted(list(mesh.faces()))

    # compute strain energy per face in the mesh
    for fkey in sorted_fkeys:

        # get bending moments stored in the mesh
        mx, my, mxy = mesh.face_attributes(fkey, names=moment_names)

        # calculate the principal bending moments
        m1a, _ = principal_stresses_and_angles(mx, my, mxy)
        _, angle1 = m1a

        # compute delta between reference vector and principal bending vector
        # TODO: will take m1 as reference. does this always hold?
        # basis vector is often the global X vector, [1, 0, 0]
        vec_ps = vf_ps[fkey]
        delta = angle1 - angle_vectors(vec_ps, basis_vector)

        # add delta to target transformation vector field
        vec = vf[fkey]
        theta = delta + angle_vectors(vec, basis_vector)

        # transform bending moments with theta
        m1, m2, m12 = transformed_stresses(mx, my, mxy, theta)

        # calculate value of structural metric
        if metric_name == "volume1":
            value = structure_func(m1, m12)
        elif metric_name == "volume2":
            value = structure_func(m2, m12)
        else:
            value = structure_func(m1, m2, m12)

        # store the bending strain energy
        data[fkey] = value

    # ==========================================================================
    # Calculate total bending strain energy
    # ==========================================================================

    total_data = data.sum()
    print("Plate {0}: {1}".format(metric_name, round(total_data, 2)))

    # ==========================================================================
    # Plot Mesh
    # ==========================================================================

    # ClusterPlotter is a custom wrapper around a COMPAS MeshPlotter
    plotter = MeshPlusPlotter(mesh, figsize=(16, 9), dpi=300)
    if draw_boundary_edges:
        plotter.draw_edges(keys=list(mesh.edges_on_boundary()))

    # draw mesh edges
    if draw_edges:
        plotter.draw_edges()

    # color up the faces of the mesh according to their cluster
    if draw_faces or draw_faces_centroids:

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

        collection.set(array=data, cmap="YlGnBu") # Spectral, BrBG, viridis, PiYG
        collection.set_linewidth(lw=0.0)

        if draw_colorbar:
            # create label for color map
            ticks = np.linspace(data.min(), data.max(), 7)
            ticks_labels = [np.round(x, 2) for x in ticks]
            extend = "both"

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
        img_name = filename.split("/")[-1] + metric_name  + "_" + vf_name + "_" + dt + ".png"
        img_path = os.path.abspath(os.path.join(DATA, "images", img_name))
        plt.tight_layout()
        plotter.save(img_path, bbox_inches='tight', pad_inches=0.0)
        print("Saved image to : {}".format(img_path))

    # show to screen
    if show_img:
        plotter.show()


if __name__ == '__main__':
    fire.Fire(plot_2d)
