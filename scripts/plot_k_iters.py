import numpy as np
import matplotlib.pyplot as plt

# plt.style.use("dark_background")

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.collections import PolyCollection

from mpl_toolkits.axes_grid1 import AxesGrid

from math import acos
from math import degrees
from math import fabs

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

from directional_clustering.geometry import clockwise
from directional_clustering.geometry import laplacian_smoothed
from directional_clustering.geometry import cosine_similarity
from directional_clustering.geometry import contour_polygons

from directional_clustering.clusters import kmeans_fit
from directional_clustering.clusters import init_kmeans_farthest
from directional_clustering.clusters import _kmeans

from directional_clustering.plotters import ClusterPlotter
from directional_clustering.plotters import rgb_colors
from directional_clustering.plotters import plot_kmeans_vectors

from compas.datastructures import Mesh
from compas.datastructures import mesh_unify_cycles

from compas.geometry import dot_vectors
from compas.geometry import scale_vector
from compas.geometry import normalize_vector
from compas.geometry import length_vector
from compas.geometry import angle_vectors
from compas.geometry import length_vector_sqrd
from compas.geometry import subtract_vectors

from compas.utilities import geometric_key

# =============================================================================
# Constants
# =============================================================================

tags = [
    "n_1",
    "n_2",
    "m_1",
    "m_2",
    "ps_1_top",
    "ps_1_bot",
    "ps_1_mid",
    "ps_2_top",
    "ps_2_bot",
    "ps_2_mid",
    "custom_1",
    "custom_2"
    ]


# HERE = "../data/json_files/two_point_wall"  # leonhardt
# HERE = "../data/json_files/wall_with_hole"  # schlaich
# HERE = "../data/json_files/cantilever_wall_3_1"  # rozvany?
# HERE = "../data/json_files/square_wall_cantilever"  # michell
# HERE = "../data/json_files/square_wall_down"  # schlaich
# HERE = "../data/json_files/perimeter_supported_slab"
HERE = "../data/json_files/four_point_slab"


tag = "m_1"
x_lim = -10.0  # faces stay if x coord of their centroid is larger than x_lim
y_lim = -10.0  # faces stay if y coord of their centroid is larger than y_lim

# =============================================================================
# Import mesh
# =============================================================================

name = HERE.split("/").pop()
mesh = Mesh.from_json(HERE + ".json")
mesh_unify_cycles(mesh)

# ==========================================================================
# Store subset attributes
# ==========================================================================

centroids = {}
vectors = {}

for fkey in mesh.faces():
    centroids[geometric_key(mesh.face_centroid(fkey))] = fkey
    vectors[fkey] = mesh.face_attribute(fkey, tag)

# ==========================================================================
# Rebuild mesh - necessary to match ordering of collection.set(array)! 
# ==========================================================================

polygons = []
for fkey in mesh.faces():
    x, y, z = mesh.face_centroid(fkey)
    if x >= x_lim and y >= y_lim:
        polygons.append(mesh.face_coordinates(fkey))

mesh = Mesh.from_polygons(polygons)
mesh_unify_cycles(mesh)

for fkey in mesh.faces():
    gkey = geometric_key(mesh.face_centroid(fkey))
    ofkey = centroids[gkey]
    vector = vectors[ofkey]
    mesh.face_attribute(fkey, tag, vector)

# =============================================================================
# Align vectors
# =============================================================================

# convexity of the resulting distribution seems to be beneficial. 
# in other words, single mode distributions produce nicer results than double
# mode. double mode distributions arise aligning with global y, whereas the good
# former one, with global x alignment.

align = True
align_ref = [1.0, 0.0, 0.0]  # global x
# align_ref = [0.0, 1.0, 0.0]  # global y

vectors = {}
for fkey in mesh.faces():
    vector = mesh.face_attribute(fkey, tag)
    if align:
         if dot_vectors(align_ref, vector) < 0:
             vector = scale_vector(vector, -1)
    vectors[fkey] = vector

# =============================================================================
# Store original vectors before smoothing
# =============================================================================

raw_values = np.zeros((mesh.number_of_faces(), 3))
for fkey, vec in vectors.items():
    raw_values[fkey,:] = vec

# =============================================================================
# Smoothen vectors
# =============================================================================

smooth_iters = 0
damping = 0.5

if smooth_iters:
    vectors = laplacian_smoothed(mesh, vectors, smooth_iters, damping)

# =============================================================================
# Cosine similarity
# =============================================================================

ref_cosim = [0.0, 1.0, 0.0]  # global y - for full systems
# ref_cosim = [1.0, 0.0, 0.0]  # global x - for symmetric partition

cosim = np.zeros(mesh.number_of_faces())
for fkey, vec in vectors.items():
    cs = cosine_similarity(ref_cosim, vec) 
    cosim[fkey] = cs

# =============================================================================
# Store vectors
# =============================================================================

values = np.zeros((mesh.number_of_faces(), 3))
normalized_values = np.zeros((mesh.number_of_faces(), 3))
squared_values = np.zeros((mesh.number_of_faces(), 3))

for fkey, vec in vectors.items():
    values[fkey,:] = vec
    normalized_values[fkey,:] = normalize_vector(vec)
    squared_values[fkey,:] = scale_vector(vec, length_vector_sqrd(vec))

# =============================================================================
# Plotting stuff
# =============================================================================

face_coords = []
for fkey in mesh.faces():
    f_coords = [point[0:2] for point in mesh.face_coordinates(fkey)]
    face_coords.append(f_coords)

# =============================================================================
# Kmeans Clustering
# =============================================================================

data_collection = {}
row_names = []
col_names = []

mode = "cosine"  # euclidean or cosine
eps = 1e-3
data = values

name = HERE.split("/")[-1]
title = "{}-{}-smooth_{}_{}_farthest".format(name, tag, smooth_iters, mode)

# fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 9), dpi=100)

fig = plt.figure(figsize=(16, 9), dpi=100)
axes = AxesGrid(
    fig,
    111,
    nrows_ncols=(2, 4),
    axes_pad=0.30,
    cbar_mode="single",
    cbar_location="left",
    cbar_pad=0.1,
    label_mode="all",
    share_all=True
)

i = 0
for n_clusters in range(3, 5):

    row_names.append("k {}".format(n_clusters))

    data_collection[i] = {}

    j = 0
    for epochs in range(5, 25, 5):
        col_names.append("{} epochs".format(epochs))

        print("***")
        print("Clustering began. k={} - epochs={}.".format(n_clusters, epochs))

        seeds = init_kmeans_farthest(data, n_clusters, mode, epochs, eps)
        km = _kmeans(data, seeds, mode, epochs, eps, early_stopping=False, verbose=True)
        labels, centers, losses = km

        print("loss kmeans", losses[-1])
        print("Clustering complete!")

# =============================================================================
# Assign clusters
# =============================================================================

        clustered_values = centers[labels]
        base = values
        target = clustered_values

# =============================================================================
# Calculate magnitudes
# =============================================================================

        magnitudes = np.zeros(mesh.number_of_faces())
        deviations = np.zeros(mesh.number_of_faces())
        losses = np.zeros(mesh.number_of_faces())

        for fkey in mesh.faces():
            # magnitudes
            vec = base[fkey, :]
            magnitudes[fkey] = np.linalg.norm(vec)

            # deviations
            deviations[fkey] = angle_vectors(base[fkey], target[fkey], deg=True)

            # losses
            losses[fkey] = length_vector_sqrd(subtract_vectors(base[fkey], target[fkey]))
        
        mse_loss = np.mean(losses)
        print("MSE Loss: {}".format(mse_loss))

# =============================================================================
# Draw cluster contours
# =============================================================================

        centers_cosim = np.array([cosine_similarity(ref_cosim, vec) for vec in centers])
        labels_cosim = np.array([cosine_similarity(ref_cosim, vec) for vec in clustered_values]) 
        cluster_contours = contour_polygons(mesh, centers_cosim, labels_cosim)

        contour_coords = []
        for contour in cluster_contours:
            contour_coords.append(contour.get("points"))

# =============================================================================
# Increase count and append data
# =============================================================================

        data_collection[i][j] = {
            "magnitudes": [magnitudes, "Blues"],
            "deviations": [deviations, "RdPu"],
            "labels": [labels, "jet"],
            "mse_loss": mse_loss,
            "contours": contour_coords
        }

        j += 1
    i += 1

# =============================================================================
# Draw subplots
# =============================================================================

to_plot = "deviations"

# for ax, col in zip(axes[0], col_names):
#    ax.set_title(col)

# for ax, row in zip(axes[:,0], row_names):
#     ax.set_ylabel(row, rotation=0, size="large", labelpad=20)

# for ax_col, col in zip(axes.axes_column, col_names):
#     for ax in ax_col:
#         ax.set_title(col)

# for ax_row, row in zip(axes.axes_row, row_names):
#     for ax in ax_row:
#         ax.set_ylabel(row, rotation=0, size="large", labelpad=20)

faces = []

for a in range(i):
    for b in range(j):

        # ax = axes[a, b]
        ax = axes.axes_row[a][b]

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

        collection = PatchCollection([Polygon(face) for face in face_coords])
        ax.add_collection(collection)

        deviations, cmap = data_collection[a][b][to_plot]

        collection.set(array=deviations, cmap=cmap)

        contours = data_collection[a][b]["contours"]

        contours = PolyCollection(contours, closed=False, linewidth=1.0, facecolors='none', edgecolor="black")
        ax.add_collection(contours)

        mse_loss = data_collection[a][b]["mse_loss"]

        xlabel = "loss: {}".format(round(mse_loss, 2))
        ax.set_xlabel(xlabel)
        
        ax.set_aspect("equal")
        ax.autoscale(tight=True)





cb = ax.cax.colorbar(collection)
cb.aspect = 50

# from matplotlib.cm import ScalarMappable

# fig.colorbar(ScalarMappable(cmap=cmap))

fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.95])
fig.suptitle(title)

# =============================================================================
# Show
# =============================================================================

plt.show()
