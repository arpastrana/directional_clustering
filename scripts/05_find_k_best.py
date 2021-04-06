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
from directional_clustering.transformations import transformed_stress_vector_fields

# plotters
from directional_clustering.plotters import MeshPlusPlotter
from directional_clustering.plotters import rgb_colors


# ==============================================================================
# Matplotlib beautification
# ==============================================================================

# plt.rcParams.update({
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})

plt.rcParams['figure.facecolor'] = 'white'
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('axes', linewidth=1.5)
plt.rc('axes', labelsize=15)
plt.rc('xtick', labelsize=20, direction="in")
plt.rc('ytick', labelsize=20, direction="in")
plt.rc('legend', fontsize=15)

# setting xtick parameters:
plt.rc('xtick.major', size=10, pad=4)
plt.rc('xtick.minor', size=5, pad=4)
plt.rc('ytick.major', size=10)
plt.rc('ytick.minor', size=5)

# ==============================================================================
# Additional Functions
# ==============================================================================

def plot_mesh_clusters(mesh, labels, draw_faces, filename, save_img):
    """
    Plot the cluster labels of a mesh.
    """

    # ClusterPlotter is a custom wrapper around a COMPAS MeshPlotter
    plotter = MeshPlusPlotter(mesh, figsize=(16, 9), dpi=100)
    plotter.draw_edges(keys=list(mesh.edges_on_boundary()))

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
    colorbar.set_label("Directional Clusters", fontsize="large")

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
                           n_init=5,
                           n_clusters_max=10,
                           eps=1,
                           iters=100,
                           tol=1e-6,
                           stop_early=False,
                           comb_vectors=False,
                           align_vectors=False,
                           alignment_ref=[1.0, 0.0, 0.0],
                           smooth_iters=0,
                           damping=0.5,
                           save_json=True,
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

    fig, ax = plt.subplots(figsize=(9, 6))

    # ==========================================================================
    # Set directory of input JSON files
    # ==========================================================================

    # Relative path to the JSON file stores the vector fields and the mesh info
    # The JSON files must be stored in the data/json_files folder

    if isinstance(filenames, str):
        filenames = [filenames]

    if isinstance(smooth_iters, int):
        smooth_iters = [smooth_iters] * len(filenames)

    for filename, smooth_iter in zip(filenames, smooth_iters):

        print("\nWorking now with: {}".format(filename))

        name_in = filename + ".json"
        json_in = os.path.abspath(os.path.join(JSON, name_in))

    # ==========================================================================
    # Import a mesh as an instance of MeshPlus
    # ==========================================================================

        mesh = MeshPlus.from_json(json_in)

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

        if smooth_iter:
            print("Smoothing vector field for {} iterations".format(smooth_iter))
            smoothen_vector_field(vectors, mesh.face_adjacency(), smooth_iter, damping)

    # ==========================================================================
    # Do K-means Clustering
    # ==========================================================================

        errors = []
        reached = False

        for i in range(1, n_clusters_max + 1):
            print()
            print("Clustering started...")
            print("Generating {} clusters".format(i))


            # perform n different initializations because of random seeding
            error_best = np.inf
            clusterer_best = None

            for _ in range(n_init):

                # Create an instance of a clustering algorithm from ClusteringFactory
                clustering_algo = ClusteringFactory.create(algo_name)
                clusterer = clustering_algo(mesh, vectors, i, iters, tol)
                clusterer.cluster()
                clustered_field = clusterer.clustered_field

    # ==========================================================================
    # Compute "loss" of clustering
    # ==========================================================================

                field_errors = np.zeros(mesh.number_of_faces())
                for fkey in mesh.faces():
                    error = distance_cosine(clustered_field.vector(fkey), vectors_raw.vector(fkey))
                    field_errors[fkey] = error
                error = np.mean(field_errors)

                # pick best contender
                if error < error_best:
                    error_best = error
                    clusterer_best = clusterer

            print("Clustering ended!")
            print("Clustered Field Mean Error (Cosine Distances) after {} init: {}".format(n_init, error_best))

    # ==========================================================================
    # Store data
    # ==========================================================================

            # record best error after n initializations
            errors.append(error_best)

            # store results in clustered_field and labels
            clustered_field = clusterer_best.clustered_field
            labels = clusterer_best.labels

    # ==========================================================================
    # Store data
    # ==========================================================================

            # plot image
            if save_img:
                plot_mesh_clusters(mesh, labels, draw_faces, filename, save_img)

            if i < 2:
                continue

            # delta_error = (errors[-2] - errors[-1]) / errors[-1]
            # print("Delta error: {}".format(delta_error))

            if fabs(error_best) <= eps:
                print("Convergence threshold of {} reached".format(eps))

                if not reached:
                    k_best = i
                    k_best_error = error_best
                    clustered_field_best = clustered_field
                    labels_best = labels
                    reached = True

                if stop_early:
                    print("Stopping early...")
                    break

    # ==========================================================================
    # Plot errors
    # ==========================================================================

        plot_label = r"\_".join(filename.split("_"))
        plot = ax.plot(errors, label=plot_label, zorder=1)
        c = plot[0].get_color()
        ax.scatter(k_best - 1, k_best_error, marker='D', s=100, color=c, zorder=2)

    # ==========================================================================
    # Variable reassignment for convencience
    # ==========================================================================

        labels = labels_best
        clustered_field = clustered_field_best

    # ==========================================================================
    # Assign cluster labels to mesh
    # ==========================================================================

        mesh.cluster_labels("cluster", labels)

    # ==========================================================================
    # Generate field orthogonal to the clustered field
    # ==========================================================================

        # add perpendicular field tha preserves magnitude
        # assumes that vector field name has format foo_bar_1 or baz_2
        vf_name_parts = vf_name.split("_")
        for idx, entry in enumerate(vf_name_parts):
            # exits at the first entry
            if entry.isnumeric():
                dir_idx = idx
                direction = entry

        n_90 = 2
        if direction == n_90:
            n_90 = 1

        vf_name_parts[dir_idx] = str(n_90)
        vf_name_90 = "_".join(vf_name_parts)

        vectors_90 = mesh.vector_field(vf_name_90)
        clustered_field_90 = VectorField()

        for fkey, _ in clustered_field.items():
            cvec_90 = cross_vectors(clustered_field[fkey], [0, 0, 1])

            scale = length_vector(vectors_90[fkey])

            if dot_vectors(cvec_90, vectors_90[fkey]):
                scale *= -1.0

            cvec_90 = scale_vector(cvec_90, scale)
            clustered_field_90.add_vector(fkey, cvec_90)

    # ==========================================================================
    # Scale fields based on stress transformations
    # ==========================================================================

        args = [mesh, (clustered_field, clustered_field_90), "bending", [1.0, 0.0, 0.0]]
        clustered_field, clustered_field_90 = transformed_stress_vector_fields(*args)

    # ==========================================================================
    # Assign clustered fields to mesh
    # ==========================================================================

        clustered_field_name = vf_name + "_k"
        mesh.vector_field(clustered_field_name, clustered_field)

        clustered_field_name_90 = vf_name_90 + "_k"
        mesh.vector_field(clustered_field_name_90, clustered_field_90)

    # ==========================================================================
    # Export new JSON file for further processing
    # ==========================================================================

        if save_json:
            name_out = "{}_k{}_{}_eps_{}_smooth_{}.json".format(filename, k_best, vf_name, eps, smooth_iter)
            json_out = os.path.abspath(os.path.join(JSON, "clustered", algo_name, name_out))
            mesh.to_json(json_out)
            print("Exported clustered vector field with mesh to: {}".format(json_out))

    # ==========================================================================
    # Customize plot
    # ==========================================================================

    plt.title(r"Best number of clusters $\hat{k}$", size=30)
    ax.grid(b=None, which='major', axis='both', linestyle='--')

    max_clusters = n_clusters_max

    ax.set_xticks(ticks=list(range(0, max_clusters)))
    ax.set_xticklabels(labels=list(range(1, max_clusters + 1)))

    ax.set_xlabel(r"Number of Clusters $k$", size=25)
    ax.set_ylabel(r"Loss $\mathcal{L}$", size=25)

    ax.legend()

    # ==========================================================================
    # Save the plot
    # ==========================================================================

    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    img_name = "k_best" + "_" + dt + ".png"
    img_path = os.path.abspath(os.path.join(DATA, "images", img_name))
    plt.tight_layout()
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0.1, dpi=600)
    print("Saved image to : {}".format(img_path))

    # ==========================================================================
    # Show the plot
    # ==========================================================================

    plt.show()


if __name__ == '__main__':
    fire.Fire(directional_clustering)
