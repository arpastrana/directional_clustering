# os
import os

# argument parsing helper
import fire

# plots and beyond
import matplotlib.pyplot as plt

# time is running out
from datetime import datetime

# hello numpy, my old friend
import numpy as np

# Library file directories
from directional_clustering import DATA
from directional_clustering import JSON

# extended version of Mesh
from directional_clustering.mesh import MeshPlus

# transformations
from directional_clustering.transformations import comb_vector_field
from directional_clustering.transformations import smoothen_vector_field

# ==============================================================================
# Matplotlib beautification
# ==============================================================================

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('axes', linewidth=1.5)
plt.rc('axes', labelsize=15)
plt.rc('xtick', labelsize=10, direction="in")
plt.rc('ytick', labelsize=10, direction="in")
plt.rc('legend', fontsize=15)

# setting xtick parameters:
plt.rc('xtick.major', size=10, pad=4)
plt.rc('xtick.minor', size=5, pad=4)
plt.rc('ytick.major', size=10)
plt.rc('ytick.minor', size=5)

# ==============================================================================
# Convenience functions
# ==============================================================================

def plot_kmeans_vectors(data, labels, normalize=False, scale_to_max=True, draw_centroids=True, flip_xy=False):
    """
    Parameters
    ----------
    data : np.array
        The vector field converted into a numpy array.
    """
    # seeds = [[11.890863876751077, 0.3545110676123977, 0.0], [2.053814486492628, -2.3074225319656225, 0.0], [1.7771751346256537, 1.9988816547970614, 0.0], [6.669670813796596, -2.87886820459424, 0.0]]
    # seeds = np.array(seeds)

    if scale_to_max or normalize:
        norm_data = np.linalg.norm(data, axis=1, keepdims=True)

    if scale_to_max:
        data = data / np.amax(norm_data)
        # seeds = seeds / np.amax(norm_data)

    if normalize:
        data = data / np.linalg.norm(data, axis=1, keepdims=True)

    n_clusters = np.amax(labels) + 1  # assumes labels are in ascending order and start at zero
    cmap = plt.cm.get_cmap('rainbow', n_clusters)  # to match that in 02_plot_2d.py

    centroids = []
    for i in range(n_clusters):

        X = data[np.nonzero(labels == i)]
        centroid = np.mean(X, axis=0)

        pos = [X[:, 0], X[:, 1]]
        centroid = [centroid[0], centroid[1]]
        centroids.append(centroid)
        axislabels = ["X direction [kN]", "Y direction [kN]"]

        if scale_to_max:
            axislabels = ["X", "Y"]

        if flip_xy:
            for mlist in (pos, centroid, axislabels):
                mlist.reverse()

        sc_color = cmap(i)
        sclabel = "Cluster {}".format(int(i + 1))
        plt.scatter(pos[0], pos[1], s=15, alpha=0.2, color=sc_color, label=sclabel)

        if draw_centroids:

            # draw lines from origin to centroid
            a = np.array([0.0, centroid[0]])  # start line at the origin
            b = np.array([0.0, centroid[1]])  # start line at the origin
            plt.plot(a, b, color="black", linewidth=0.5, ls="--")  # plot line

            # draw origin as point
            plt.scatter(0.0, 0.0, marker="o", color="black", alpha=1.0, s=25)
            # draw centroid as point
            plt.scatter(centroid[0], centroid[1], marker="o", color=sc_color, edgecolors="black", alpha=1.0, s=100)

            # plt.scatter(seeds[:, 0], seeds[:, 1], marker="s", color="black", s=100)
            # plt.plot([seeds, centroid, color='black', lw="--")

            # width = 0.001
            # plt.arrow(0, 0, centroid[0], centroid[1],
            #           color="dimgray",
            #           ls="-",
            #           width=width,
            #           head_width=20*width,
            #           head_length=20*width)

    # plt.scatter(seeds[:, 0], seeds[:, 1], marker="x", color="black", s=100)
    # for seed, centroid in zip(seeds, centroids):
    #     plt.plot(np.array([seed[0], centroid[0]]), np.array([seed[1], centroid[1]]), color='black', ls="--")

    # plt.grid(b=None, which='major', axis='both', linestyle='--', linewidth=0.5)
    # plt.xlabel(axislabels[0], size=15)
    # plt.ylabel(axislabels[1], size=15)

    plt.legend(loc="best", fontsize="xx-small", markerscale=3)


# ==============================================================================
# Main course
# ==============================================================================

def plot_vectors_2d(filename,
                    vf_name,
                    comb=False,
                    smooth_iters=0,
                    draw_centroids=True,
                    normalize=False,
                    scale_to_max=False,
                    flip_xy=False,
                    show_img=True,
                    save_img=False,
                    figsize=(6, 8)):
    """
    Make a plot of a vector field and its clusters. No mesh.
    """

    # load a mesh from a JSON file
    name_in = filename + ".json"
    json_in = os.path.abspath(os.path.join(JSON, "clustered", name_in))

    mesh = MeshPlus.from_json(json_in)

    # fetch labels
    labels = mesh.cluster_labels("cluster")
    labels_array = np.zeros(mesh.number_of_faces(), dtype=np.int64)
    for fkey, label in labels.items():
        labels_array[fkey] = int(label)

    # get vector field
    vf = mesh.vector_field(vf_name)

    # Comb the vector field -- remember the hair ball theorem (seams exist)
    if comb:
        vf = comb_vector_field(vf, mesh)

    # smooth vector field
    if smooth_iters:
        smoothen_vector_field(vf, mesh.face_adjacency(), smooth_iters, 0.5)

    vectors_array = np.zeros((mesh.number_of_faces(), 3))  # assumes vectors live in 3d
    for fkey in mesh.faces():
        vector = vf[fkey]
        vectors_array[fkey, :] = vector

    # plot?
    plt.figure(figsize=figsize)
    plot_kmeans_vectors(vectors_array,
                        labels_array,
                        normalize=normalize,
                        scale_to_max=scale_to_max,
                        draw_centroids=draw_centroids,
                        flip_xy=flip_xy)

    # customize plot
    plt.title("{Cluster Assignments on Vector Field}")

    # save
    if save_img:
        dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        img_name = "vf_clusters" + "_" + dt + ".png"
        img_path = os.path.abspath(os.path.join(DATA, "images", img_name))
        plt.tight_layout()
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0.1, dpi=600)
        print("Saved image to : {}".format(img_path))

    # show plot
    if show_img:
        plt.show()

# ==============================================================================
# Executable
# ==============================================================================


if __name__ == '__main__':
    fire.Fire(plot_vectors_2d)
