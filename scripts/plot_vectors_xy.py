import numpy as np
import matplotlib.pyplot as plt

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

from directional_clustering.plotters import ClusterPlotter
from directional_clustering.plotters import rgb_colors
from directional_clustering.plotters import plot_colored_vectors

from compas.datastructures import Mesh
from compas.datastructures import mesh_unify_cycles

from compas.geometry import dot_vectors
from compas.geometry import scale_vector
from compas.geometry import normalize_vector
from compas.geometry import length_vector
from compas.geometry import angle_vectors

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


# HERE = "../data/json_files/two_point_wall"  # leonhardt
# HERE = "../data/json_files/wall_with_hole"  # schlaich
# HERE = "../data/json_files/cantilever_wall_3_1"  # rozvany?
# HERE = "../data/json_files/square_wall_cantilever"  # michell
# HERE = "../data/json_files/square_wall_down_res_005"  # schlaich
HERE = "../data/json_files/four_point_slab"  # schlaich


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

align = True
align_ref = [1.0, 0.0, 0.0]  # global x

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
normalize = True
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

    # helps clustering in a good way
    # square cosim values and keep sign
    # s = np.sign(cs)
    # cs = s * (cs ** 2)

    cosim[fkey] = cs

# =============================================================================
# Plot cosim as heights
# =============================================================================

plot_points_3d = False

if plot_points_3d:

    # plot in 3d
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

for fkey, vec in vectors.items():
    smoothed_values[fkey,:] = vec
    values[fkey,:] = normalize_vector(vec)
    values[fkey,:] = vec

# =============================================================================
# Kmeans Clustering
# =============================================================================

n_clusters = 10
do_kmeans = True

if do_kmeans:
    print("Clustering started...")

    km = KMeans(n_clusters=n_clusters)
    km.fit(values)

    labels = km.labels_
    centers = km.cluster_centers_

    # suffers from initialization!
    km = kmeans_fit(values, n_clusters, dist="cosim", epochs=20, eps=1e-3, early_stopping=True, verbose=True)
    labels, centers, losses = km
    print("loss kmeans", losses[-1])

    clustered_values = centers[labels]

    print("Clustering ended!")

# =============================================================================
# Recalibrate centers to account for raw magnitudes
# =============================================================================

    base = smoothed_values

    recalibrated_values = np.zeros(clustered_values.shape)
    for i in range(n_clusters):
        face_indices = np.nonzero(labels==i)
        new_vector = np.mean(base[face_indices], axis=0)
        recalibrated_values[face_indices] = new_vector
    
# =============================================================================
# Spectral Clustering - (deprecated)
# =============================================================================

# n_clusters = 3
# do_sc = False

# if do_sc:  # problem with SC is that it is very slow with n_clusters>=5
#     sc = SpectralClustering(n_clusters, eigen_solver="arpack", affinity="rbf", n_neighbors=5, assign_labels="discretize", random_state=0)

#     sc.fit(values)
#     labels = sc.labels_

# =============================================================================
# Calculate magnitudes
# =============================================================================

base = smoothed_values

magnitudes = np.zeros(mesh.number_of_faces())
for fkey in mesh.faces():
    vec = base[fkey, :]
    magnitudes[fkey] = np.linalg.norm(vec)

# =============================================================================
# Set clustered vectors as attributes
# =============================================================================

base = clustered_values
name = tag + "_k_{}".format(n_clusters)

for fkey in mesh.faces():
    vec = base[fkey, :].tolist()
    mesh.face_attribute(key=fkey, name=name, value=vec)

# =============================================================================
# Calculate resulting deviation
# =============================================================================

base = values
target = recalibrated_values

deviations = np.zeros(mesh.number_of_faces())
for fkey in mesh.faces():
    # deviations[fkey] = fabs(cosine_similarity(base[fkey], target[fkey]))
    deviations[fkey] = angle_vectors(base[fkey], target[fkey], deg=True)

# deviations = 1.0 - deviations  # invert results

print("Loss: {}".format(np.sum(deviations)))

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
    plotter.draw_vector_field_array(target, (0, 0, 0), True, 0.07, width=1.0)
    # plotter.draw_vector_field_array(base, (50, 50, 50), True, 0.07, width=0.5)

# =============================================================================
# Data to color
# =============================================================================

dataset = "deviations"

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

# plt.colormaps()

collection.set(array=data, cmap=cmap)  # deviations
colorbar = plotter.figure.colorbar(collection)

# collection.set_clim(vmin=round(min_cosim, 2), vmax=round(max_cosim, 2))
# ticks = [min_cosim] + colorbar.get_ticks().tolist() + [max_cosim]
# colorbar.set_ticks([round(t, 2) for t in ticks])

# =============================================================================
# Draw cluster contours
# =============================================================================

draw_cluster_contours = True

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
# Show
# =============================================================================

export_json = False

if export_json:
    out = HERE + "_k_{}.json".format(n_clusters)
    mesh.to_json(out)
    print("Exported mesh to: {}".format(out))
