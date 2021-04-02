# mathematics
from math import fabs

# os
import os

# argument parsing helper
import fire

# good ol' numpy
import numpy as np

# pyplot can't miss the party
import matplotlib.pyplot as plt

# time is running out
from datetime import datetime

# geometry helpers
from compas.geometry import cross_vectors
from compas.geometry import length_vector
from compas.geometry import scale_vector
from compas.geometry import dot_vectors

# JSON file directory
from directional_clustering import DATA
from directional_clustering import JSON

# extended version of Mesh
from directional_clustering.mesh import MeshPlus

# clustering algorithms factory
from directional_clustering.clustering import ClusteringFactory
from directional_clustering.clustering import distance_cosine

# vector field
from directional_clustering.fields import VectorField

# transformations
from directional_clustering.transformations import align_vector_field
from directional_clustering.transformations import smoothen_vector_field
from directional_clustering.transformations import comb_vector_field

# plotters
from directional_clustering.plotters import MeshPlusPlotter
from directional_clustering.plotters import rgb_colors


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})


def plot_mesh_clusters(mesh, labels, draw_faces, filename, save_img):
    # ClusterPlotter is a custom wrapper around a COMPAS MeshPlotter
    plotter = MeshPlusPlotter(mesh, figsize=(16, 9), dpi=100)
    plotter.draw_edges(keys=list(mesh.edges_on_boundary()))
    # face_colors = rgb_colors(labels)
    # plotter.draw_faces(facecolor=face_colors)

    data = np.zeros(mesh.number_of_faces())
    sorted_fkeys = sorted(list(mesh.faces()))
    n_clusters = len(set(labels.values()))

    for fkey, label in labels.items():
        data[fkey] = label

    cmap = plt.cm.get_cmap('rainbow', n_clusters)
    ticks = np.linspace(0, n_clusters - 1, n_clusters + 1) + 0.5 * (n_clusters - 1)/n_clusters
    ticks = ticks[:-1]
    ticks_labels = list(range(1, n_clusters + 1))
    extend = "neither"

    if draw_faces:
        collection = plotter.draw_faces(keys=sorted_fkeys)

    else:
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
    colorbar.set_label("Directional Clusters", fontsize="xx-large")

    if save_img:
        dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        img_name = filename.split("/")[-1] + "_" + dt + ".png"
        img_path = os.path.abspath(os.path.join(DATA, "images", img_name))
        plt.tight_layout()
        plotter.save(img_path, bbox_inches='tight', pad_inches=0)
        print("Saved image to : {}".format(img_path))

    return plotter

# ==============================================================================
# Main function: directional_clustering
# ==============================================================================

def directional_clustering(filenames,
                           vf_name,
                           algo_name="cosine_kmeans",
                           n_clusters_max=10,
                           iters=100,
                           tol=1e-6,
                           comb_vectors=False,
                           align_vectors=False,
                           alignment_ref=[1.0, 0.0, 0.0],
                           smooth_iters=0,
                           damping=0.5,
                           stop_early=False,
                           eps=1,
                           save_img=False,
                           draw_faces=True):
    """
    Clusters a vector field that has been defined on a mesh. Exports a JSON file.

    Parameters
    ----------
    filename : `str`
        The name of the JSON file that encodes a mesh.
        All JSON files must reside in this repo's data/json folder.

    algo_name : `str`
        The name of the algorithm to cluster the vector field.
        \nSupported options are `cosine_kmeans` and `variational_kmeans`.

    n_clusters : `int`
        The number of clusters to generate.

    iters : `int`
        The number of iterations to run the clustering algorithm for.

    tol : `float`
        A small threshold value that marks clustering convergence.
        \nDefaults to 1e-6.

    align_vectors : `bool`
        Flag to align vectors relative to a reference vector.
        \nDefaults to False.

    alignment_ref : `list` of `float`
        The reference vector for alignment.
        \nDefaults to [1.0, 0.0, 0.0].

    smooth_iters : `int`
        The number iterations of Laplacian smoothing on the vector field.
        \nIf set to 0, no smoothing will take place.
        Defaults to 0.

    damping : `float`
        A value between 0.0 and 1.0 to control the intensity of the smoothing.
        \nZero technically means no smoothing. One means maximum smoothing.
        Defaults to 0.5.
    """

    # ==========================================================================
    # Make a plot
    # ==========================================================================

    fig, ax = plt.subplots()

    # ==========================================================================
    # Set directory of input JSON files
    # ==========================================================================

    # Relative path to the JSON file stores the vector fields and the mesh info
    # The JSON files must be stored in the data/json_files folder

    if isinstance(filenames, str):
        filenames = [filenames]

    for filename in filenames:

        name_in = filename + ".json"
        json_in = os.path.abspath(os.path.join(JSON, name_in))

        # ==========================================================================
        # Import a mesh as an instance of MeshPlus
        # ==========================================================================

        mesh = MeshPlus.from_json(json_in)

        # ==========================================================================
        # Search for supported vector field attributes and take one choice from user
        # ==========================================================================

        # supported vector field attributes
        # available_vf = mesh.vector_fields()
        # print("Avaliable vector fields on the mesh are:\n", available_vf)

        # # the name of the vector field to cluster.
        # while True:
        #     vf_name = input("Please choose one vector field to cluster:")
        #     if vf_name in available_vf:
        #         break
        #     else:
        #         print("This vector field is not available. Please try again.")

        # ==========================================================================
        # Extract vector field from mesh for clustering
        # ==========================================================================

        vectors = mesh.vector_field(vf_name)
        vectors_raw = mesh.vector_field(vf_name)

        # ==========================================================================
        # Align vector field to a reference vector
        # ==========================================================================

        if align_vectors:
            align_vector_field(vectors, alignment_ref)

        # ==========================================================================
        # Comb the vector field -- remember the hair ball theorem (seams exist)
        # ==========================================================================

        if comb_vectors:
            vectors = comb_vector_field(vectors, mesh)

        # ==========================================================================
        # Apply smoothing to the vector field
        # ==========================================================================

        if smooth_iters:
            smoothen_vector_field(vectors, mesh.face_adjacency(), smooth_iters, damping)

        # ==========================================================================
        # Do K-means Clustering
        # ==========================================================================

        errors = []
        marker = []
        markersize = [0.01] * n_clusters_max
        reached = False

        for i in range(1, n_clusters_max + 1):
            print()
            print("Clustering started...")
            print("Generating {} clusters".format(i))
            # Create an instance of a clustering algorithm from ClusteringFactory
            clustering_algo = ClusteringFactory.create(algo_name)
            clusterer = clustering_algo(mesh, vectors, i, iters, tol)
            clusterer.cluster()
            print("Clustering ended!")

            # store results in clustered_field and labels
            clustered_field = clusterer.clustered_field
            labels = clusterer.labels

            # ==========================================================================
            # Compute "loss" of clustering
            # ==========================================================================

            field_errors = np.zeros(mesh.number_of_faces())
            for fkey in mesh.faces():
                error = distance_cosine(clustered_field.vector(fkey), vectors_raw.vector(fkey))
                field_errors[fkey] = error

            error = np.mean(field_errors)
            print("Clustered Field Mean Error (Cosine Distances): {}".format(error))

            errors.append(error)

            # plot image
            if save_img:
                plot_mesh_clusters(mesh, labels, draw_faces, filename, save_img)

            if i < 2:
                continue

            delta_error = (errors[-2] - errors[-1]) / errors[-1]
            print("Delta error: {}".format(delta_error))

            if fabs(delta_error) <= eps:
                print("Convergence threshold of {} reached".format(eps))

                if not reached:
                    k_best = i
                    k_best_error = error
                    reached = True

                if stop_early:
                    print("Stopping early...")
                    break

        # ==========================================================================
        # Plot errors
        # ==========================================================================

        plot = ax.plot(errors, label=filename, zorder=1)

        c = plot[0].get_color()
        ax.scatter(k_best - 1, k_best_error, marker='D', color=c, zorder=2)

    # ==========================================================================
    # Customize plot
    # ==========================================================================

    ax.grid(b=None, which='major', axis='both', linestyle='--')

    max_clusters = n_clusters_max

    ax.set_xticks(ticks=list(range(0, max_clusters)))
    ax.set_xticklabels(labels=list(range(1, max_clusters + 1)))

    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel(r"Loss - $\mathcal{L}$")

    ax.legend()

    # ==========================================================================
    # Show the plot
    # ==========================================================================

    plt.show()


if __name__ == '__main__':
    fire.Fire(directional_clustering)
