import numpy as np
import matplotlib.pyplot as plt

from math import acos
from math import degrees
from math import fabs
from math import radians

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
from compas.geometry import rotate_points
from compas.geometry import cross_vectors

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
# HERE = "../data/json_files/square_wall_cantilever"  # michell
# HERE = "../data/json_files/square_wall_down_res_005"  # schlaich
# HERE = "../data/json_files/perimeter_supported_slab"  # schlaich
# HERE = "../data/json_files/perimeter_supported_vault_z500mm"  #vault
 
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
# Plot vectors 2d
# =============================================================================

plot_vectors_2d = False
normalize = False
rescale = False
vec_scale = 1.0  # for rescaling, max length

if plot_vectors_2d:
    lengths = [length_vector(vec) for k, vec in vectors.items()]
    max_length = max(lengths)
    min_length = min(lengths)

    x = np.zeros(mesh.number_of_faces())
    y = np.zeros(mesh.number_of_faces())

    for fkey, vector in vectors.items():

        if normalize:
            vector = normalize_vector(vector)

        if rescale:
            length = length_vector(vector)
            centered_val = (length - min_length)  # in case min is not 0
            ratio = centered_val / max_length
            vector = scale_vector(vector, ratio * vec_scale)

        x[fkey] = vector[0]
        y[fkey] = vector[1]

    plt.scatter(x, y, alpha=0.3)
    plt.grid(b=None, which='major', axis='both', linestyle='--')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

# =============================================================================
# Store original vectors before smoothing and normalization
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
# Plot cosim as heights
# =============================================================================

plot_points_3d = False

if plot_points_3d:

    D = np.zeros((mesh.number_of_faces(), 3))
    for fkey in mesh.faces():
        x, y, z = mesh.face_centroid(fkey)
        
        D[fkey, 0] = x
        D[fkey, 1] = y
        D[fkey, 2] = cosim[fkey]
    
    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection="3d")

    x = D[:,0]
    y = D[:,1]
    z = D[:,2]

    ax.scatter3D(x, y, z, c=z, cmap="jet")
    ax.view_init(70, -65)

    plt.show()

# =============================================================================
# Normalize vectors
# =============================================================================

smoothed_values = np.zeros((mesh.number_of_faces(), 3))
values = np.zeros((mesh.number_of_faces(), 3))
normalized_values = np.zeros((mesh.number_of_faces(), 3))
squared_values = np.zeros((mesh.number_of_faces(), 3))

for fkey, vec in vectors.items():
    smoothed_values[fkey,:] = vec
    values[fkey,:] = vec
    normalized_values[fkey,:] = normalize_vector(vec)
    squared_values[fkey,:] = scale_vector(vec, length_vector_sqrd(vec))

# =============================================================================
# Kmeans Clustering
# =============================================================================

n_clusters = 5
do_kmeans = True
data = values

if do_kmeans:
    print("Clustering started...")

    # furthest seed initialization
    mode = "cosine"  # euclidean or cosine
    eps = 1e-3
    epochs = 100
    seeds = init_kmeans_farthest(data, n_clusters, mode, epochs, eps)
    epochs = 100
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
# Set clustered vectors as attributes
# =============================================================================

base = clustered_values
name_1 = tag + "_k".format(n_clusters)
name_2 = tag[:-1] + "2_k".format(n_clusters)
recalibrated_perp = np.zeros(clustered_values.shape)

for fkey in mesh.faces():
    vec = base[fkey, :].tolist()
    mesh.face_attribute(key=fkey, name=name_1, value=vec)

    vec_2 = cross_vectors(vec, [0.0, 0.0, 1.0])
    # vec_2 = rotate_points([vec], radians(90.0), origin=mesh.face_centroid(fkey))
    # vec_2 = vec_2[0]
    mesh.face_attribute(key=fkey, name=name_2, value=vec_2)
    recalibrated_perp[fkey, :] = vec_2

# =============================================================================
# Calculate resulting deviation
# =============================================================================

base = values
target = recalibrated_values

deviations = np.zeros(mesh.number_of_faces())
for fkey in mesh.faces():
    deviations[fkey] = angle_vectors(base[fkey], target[fkey], deg=True)

# =============================================================================
# Compute MSE Loss
# =============================================================================

losses = np.zeros(mesh.number_of_faces())
for fkey in mesh.faces():    
    losses[fkey] = length_vector_sqrd(subtract_vectors(base[fkey], target[fkey]))

print("MSE Loss: {}".format(np.mean(deviations)))

# =============================================================================
# Draw kmeans vectors
# =============================================================================

draw_kmeans_vectors = False

if draw_kmeans_vectors:
    plot_kmeans_vectors(values, labels, centers, normalize=False, draw_centroids=True)

# =============================================================================
# Plotter
# =============================================================================

plotter = ClusterPlotter(mesh, figsize=(12, 9))
plotter.draw_edges(keys=list(mesh.edges_on_boundary()))

# =============================================================================
# Draw vector fields
# =============================================================================

draw_vector_fields = True

if draw_vector_fields:
    # plotter.draw_vector_field_array(target, (0, 0, 0), True, 0.07, width=1.0)
    # plotter.draw_vector_field_array(base, (50, 50, 50), True, 0.07, width=0.5)

    plotter.draw_vector_field_array(target, (0, 0, 0), True, 0.05, width=0.5)
    plotter.draw_vector_field_array(recalibrated_perp, (0, 0, 0), True, 0.05, width=0.5)

# =============================================================================
# Data to color
# =============================================================================

dataset = "labels"

data_collection = {
    "labels": {"values": labels, "cmap": "jet"},
    "deviations": {"values": deviations, "cmap": "RdPu"},  # also cmap: Grays
    "cosim": {"values": cosim, "cmap": "RdBu"},  # also cmap: Spectral
    "magnitudes": {"values": magnitudes, "cmap": "Blues"}
}

data = data_collection[dataset]["values"]
cmap = data_collection[dataset]["cmap"]

# =============================================================================
# Color Faces
# =============================================================================

collection = plotter.draw_faces()
collection.set(array=data, cmap=cmap)
colorbar = plotter.figure.colorbar(collection)

# =============================================================================
# Draw cluster contours
# =============================================================================

draw_cluster_contours = False

if draw_cluster_contours:
    centers_cosim = np.array([cosine_similarity(ref_cosim, vec) for vec in centers])
    labels_cosim = np.array([cosine_similarity(ref_cosim, vec) for vec in clustered_values])

    cluster_contours = contour_polygons(mesh, centers_cosim, labels_cosim)
    plotter.draw_polylines(cluster_contours)

# =============================================================================
# Show
# =============================================================================

plotter.show()

# =============================================================================
# Export json
# =============================================================================

export_json = True

if export_json:
    out = HERE + "_k_{}.json".format(n_clusters)
    mesh.to_json(out)
    print("Exported mesh to: {}".format(out))
