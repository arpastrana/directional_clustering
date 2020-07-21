import numpy as np
import matplotlib.pyplot as plt

from math import degrees
from math import fabs

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

from directional_clustering.geometry import clockwise
from directional_clustering.geometry import smoothed_angles
from directional_clustering.geometry import laplacian_smoothed
from directional_clustering.geometry import cosine_similarity

from directional_clustering.clusters import faces_angles
from directional_clustering.clusters import faces_labels
from directional_clustering.clusters import kmeans_clustering
from directional_clustering.clusters import kmeans_errors
from directional_clustering.clusters import faces_clustered_field

from directional_clustering.plotters import ClusterPlotter
from directional_clustering.plotters import rgb_colors
from directional_clustering.plotters import plot_colored_vectors

from compas.datastructures import Mesh
from compas.datastructures import mesh_unify_cycles

from compas.geometry import dot_vectors
from compas.geometry import scale_vector
from compas.geometry import normalize_vector
from compas.geometry import length_vector

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

HERE = "../data/json_files/four_point_slab"

tag = "m_1"
x_lim = -10.0  # faces stay if x coord of their centroid is larger than x_lim

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
# Rebuild mesh
# ==========================================================================

polygons = [mesh.face_coordinates(fkey) for fkey in mesh.faces() if mesh.face_centroid(fkey)[0] >= x_lim]
mesh = Mesh.from_polygons(polygons)
mesh_unify_cycles(mesh)

for fkey in mesh.faces():
    gkey = geometric_key(mesh.face_centroid(fkey))
    ofkey = centroids[gkey]
    vector = vectors[ofkey]
    mesh.face_attribute(fkey, tag, vector)

# =============================================================================
# Process PS vectors
# =============================================================================

align = True
# align_ref = [0.0, 1.0, 0.0]
align_ref = [1.0, 0.0, 0.0]
normalize = False

rescale = False
vec_scale = 1.0  # for rescaling, max length

# =============================================================================
# Align vectors
# =============================================================================

vectors = {}
for fkey in mesh.faces():
    vector = mesh.face_attribute(fkey, tag) 
    if align:
        if dot_vectors(align_ref, vector) < 0:
            vector = scale_vector(vector, -1)
    if normalize:
        vector = normalize_vector(vector)
    vectors[fkey] = vector

# =============================================================================
# Scale and normalize vectors
# =============================================================================

plot_vectors_2d = False

if plot_vectors_2d:
    lengths = [length_vector(vec) for k, vec in vectors.items()]
    max_length = max(lengths)
    min_length = min(lengths)

    x = np.zeros(mesh.number_of_faces())
    y = np.zeros(mesh.number_of_faces())

    for fkey, vector in vectors.items():

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
# Cosine similarity
# =============================================================================

ref_cosim = [0.0, 1.0, 0.0]
# ref_cosim = [1.0, 0.0, 0.0]

cosim = {}
for fkey, vector in vectors.items():
    # cosim[fkey] = cosine_similarity(ref_cosim, vector)
    cosim[fkey] = normalize_vector(vector)  # cluster based on normalized and aligned vectors

# =============================================================================
# Smoothen vectors
# =============================================================================

smooth_iters = 0
damping = 0.5

if smooth_iters:
    cosim = laplacian_smoothed(mesh, cosim, smooth_iters, damping)

# min_cosim = min(cosim.values())
# max_cosim = max(cosim.values())

# =============================================================================
# Print stuff
# =============================================================================

# print("max cosim: {}".format(max_cosim))
# print("min cosim: {}".format(min_cosim))

# =============================================================================
# Dict to array
# =============================================================================

# values = np.zeros((mesh.number_of_faces(), 1))
# for fkey, cs in cosim.items():
#     values[fkey] = cs

values = np.zeros((mesh.number_of_faces(), 3))
for fkey, cs in cosim.items():
    values[fkey,:] = normalize_vector(cs)

# =============================================================================
# # Plot 3d - Data as Z
# # =============================================================================

# D = np.zeros((mesh.number_of_faces(), 3))

# for fkey in mesh.faces():
#     x, y, z = mesh.face_centroid(fkey)
    
#     D[fkey, 0] = x
#     D[fkey, 1] = y
#     D[fkey, 2] = values[fkey, :]

# =============================================================================
# Kmeans Clustering
# =============================================================================

n_clusters = 5
do_kmeans = True

if do_kmeans:

    km = KMeans(n_clusters=n_clusters, init="random", random_state=0)
    km.fit(values)

    labels = km.labels_
    centers = km.cluster_centers_

# =============================================================================
# Spectral Clustering
# =============================================================================

n_clusters = 4
do_sc = False

if do_sc:  # problem with SC is that it is very slow with n_clusters>=5
    sc = SpectralClustering(n_clusters, eigen_solver="arpack", affinity="rbf", n_neighbors=5, assign_labels="discretize", random_state=0)

    sc.fit(values)
    labels = sc.labels_
    print("labels shape", labels.shape)

# =============================================================================
# Spectral Clustering
# =============================================================================

# =============================================================================
# Plotter
# =============================================================================

plotter = ClusterPlotter(mesh, figsize=(12, 9))
plotter.draw_edges(keys=list(mesh.edges_on_boundary()))
plotter.draw_vector_field(tag, (0, 0, 0), True, 0.05, 1.0)

# =============================================================================
# Color Faces
# =============================================================================

collection = plotter.draw_faces()

values = labels

"""
Divergent colormaps
luminance highest at midpoint

RdBu      red, white, blue (ok)
RdYlBu    red, yellow, blue (ok)
RdYlGn    red, yellow, green (ok)
Spectral  red, orange, yellow, green, blue (ok)
"""

collection.set(array=values, cmap='jet')
# collection.set_clim(vmin=round(min_cosim, 2), vmax=round(max_cosim, 2))
colorbar = plotter.figure.colorbar(collection)
# ticks = [min_cosim] + colorbar.get_ticks().tolist() + [max_cosim]
# colorbar.set_ticks([round(t, 2) for t in ticks])

# =============================================================================
# Show
# =============================================================================

plotter.show()
