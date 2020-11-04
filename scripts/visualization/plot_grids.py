import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.collections import PolyCollection

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
# HERE = "../data/json_files/four_point_slab"
HERE = "../../data/json_files/perimeter_supported_vault_z500mm"


tag = "ps_1_mid"
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
# Kmeans Clustering
# =============================================================================

n_clusters = 3

data = values

print("Clustering started...")

# furthest seed initialization
mode = "cosine"  # euclidean or cosine
eps = 1e-3
epochs = 50
seeds = init_kmeans_farthest(data, n_clusters, mode, epochs, eps)
km = _kmeans(data, seeds, mode, epochs, eps, early_stopping=False, verbose=True)

labels, centers, losses = km

print("loss kmeans", losses[-1])
print("Clustering ended!")

# =============================================================================
# Assign clusters
# =============================================================================

clustered_values = centers[labels]

# =============================================================================
# Recalibrate centers to account for raw magnitudes
# =============================================================================

base = values

recalibrated_values = np.zeros(clustered_values.shape)
for i in range(n_clusters):
    face_indices = np.nonzero(labels==i)
    new_vector = np.mean(base[face_indices], axis=0)
    recalibrated_values[face_indices] = new_vector

# =============================================================================
# Calculate magnitudes
# =============================================================================

base = values

magnitudes = np.zeros(mesh.number_of_faces())
for fkey in mesh.faces():
    vec = base[fkey, :]
    magnitudes[fkey] = np.linalg.norm(vec)

# =============================================================================
# Calculate resulting deviation
# =============================================================================

base = values
target = clustered_values

deviations = np.zeros(mesh.number_of_faces())
for fkey in mesh.faces():
    deviations[fkey] = angle_vectors(base[fkey], target[fkey], deg=True)

# =============================================================================
# Compute MSE Loss
# =============================================================================

losses = np.zeros(mesh.number_of_faces())
for fkey in mesh.faces():    
    losses[fkey] = length_vector_sqrd(subtract_vectors(base[fkey], target[fkey]))
mse_loss = np.mean(deviations)

print("MSE Loss: {}".format(mse_loss))

# =============================================================================
# Data to color
# =============================================================================

data_collection = {
    "labels": {"values": labels, "cmap": "jet", "bins": True},
    "deviations_deg": {"values": deviations, "cmap": "RdPu"},
    "cosine similarity to X": {"values": cosim, "cmap": "RdBu"},
    "magnitudes": {"values": magnitudes, "cmap": "Blues"}
}

# =============================================================================
# Draw cluster contours
# =============================================================================

centers_cosim = np.array([cosine_similarity(ref_cosim, vec) for vec in centers])
labels_cosim = np.array([cosine_similarity(ref_cosim, vec) for vec in clustered_values]) 
cluster_contours = contour_polygons(mesh, centers_cosim, labels_cosim)

# =============================================================================
# Draw subplots
# =============================================================================

face_coords = []
for fkey in mesh.faces():
    f_coords = [point[0:2] for point in mesh.face_coordinates(fkey)]
    face_coords.append(f_coords)

contour_coords = []
for contour in cluster_contours:
    contour_coords.append(contour.get("points"))

fig = plt.figure(figsize=(16, 9), dpi=100)

data_keys = list(data_collection.keys())

for i in range(1, len(data_collection) + 1):
    ax = fig.add_subplot(1, 4, i)

    ax.set_xticks([])
    ax.set_yticks([])

    collection = PatchCollection([Polygon(face) for face in face_coords])
    ax.add_collection(collection)

    dataset = data_keys[i-1]
    data = data_collection[dataset]["values"]
    cmap = data_collection[dataset]["cmap"]
    bins = data_collection[dataset].get("bins")

    if bins:
        bins = np.amax(data) + 1
        ticks = range(bins)

    cmap = plt.get_cmap(cmap, lut=bins)

    collection.set(array=data, cmap=cmap, edgecolor=None)
    
    if bins:
        plt.colorbar(collection, label=dataset, ticks=ticks, aspect=50)
    else: 
        plt.colorbar(collection, label=dataset, aspect=50)

    # add contour plots
    # contours = PolyCollection(contour_coords, closed=False, linewidth=1.0, facecolors='none')
    # ax.add_collection(contours)

    title = "K: {}/ Epochs: {} / MSE: {}".format(n_clusters, epochs, round(mse_loss, 2))

    ax.set_title(title)
    ax.set_aspect("equal")

    ax.set_frame_on(False)
    ax.autoscale()

plt.tight_layout()
plt.show()

# =============================================================================
# Show
# =============================================================================

plt.show()

