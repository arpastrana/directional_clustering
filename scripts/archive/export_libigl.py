import numpy as np
import matplotlib.pyplot as plt

# plt.style.use("dark_background")

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.collections import PolyCollection

from mpl_toolkits.axes_grid1 import AxesGrid

# import mpl_toolkits

# = mpl_toolkits.legacy_colorbar

# legacy_colorbar.rcParam = False

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


THERE = "/Users/arpj/code/libraries/libigl/tutorial/508_ARP_MIQ/"

# HERE = "../data/json_files/two_point_wall"  # leonhardt
# HERE = "../data/json_files/wall_with_hole"  # schlaich
# HERE = "../data/json_files/cantilever_wall_3_1"  # rozvany?
# HERE = "../data/json_files/square_wall_cantilever"  # michell
# HERE = "../data/json_files/square_wall_down"  # schlaich
# HERE = "../data/json_files/perimeter_supported_slab"

HERE = "../data/json_files/four_point_slab"
# HERE = "../data/json_files/four_point_slab_k_7"
# HERE = "../data/json_files/perimeter_supported_slab_k_5"
# HERE = "../data/json_files/perimeter_supported_slab"
HERE = "../data/json_files/perimeter_supported_vault_z500mm_k_3"  #vault

tag = "n_1_k"
tag_2 = "n_2_k"

# tag = "m_1_k"
# tag_2 = "m_2_k"

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
vectors_2 = {}

for fkey in mesh.faces():
    centroids[geometric_key(mesh.face_centroid(fkey))] = fkey
    vectors[fkey] = mesh.face_attribute(fkey, tag)
    vectors_2[fkey] = mesh.face_attribute(fkey, tag_2)

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
    vector_2 = vectors_2[ofkey]
    mesh.face_attribute(fkey, tag, vector)
    mesh.face_attribute(fkey, tag_2, vector_2)

# =============================================================================
# Export vertices and faces
# =============================================================================

vertices, faces = mesh.to_vertices_and_faces()

V = np.array(vertices)
print("V shape: ", V.shape)
print("V first row: {}".format(V[0,:]))
print("V last row: {}".format(V[-1,:]))

F = np.array(faces)
print("F shape: ", F.shape)
print("F first row: {}".format(F[0,:]))
print("F last row: {}".format(F[-1,:]))

np.savetxt(THERE + "vertices.txt", V, fmt="%1.6f", delimiter=" ", encoding=None)
np.savetxt(THERE + "faces.txt", F, fmt="%d", delimiter=" ", encoding=None)

# # =============================================================================
# # Export edges on boundary
# # =============================================================================

# E = np.array(mesh.edges_on_boundary())
# print("E shape: ", E.shape)
# print("E first row: {}".format(E[0,:]))
# print("E last row: {}".format(E[-1,:]))

# np.savetxt(THERE + "edges_boundary.txt", E, fmt="%d", delimiter=" ", encoding=None)

# # =============================================================================
# # Export vertices on boundary
# # =============================================================================

# B = np.array(mesh.vertices_on_boundary())
# print("B shape: ", B.shape)
# print("B first row: {}".format(B[0]))
# print("B last row: {}".format(B[-1]))

# np.savetxt(THERE + "vertices_boundary.txt", E, fmt="%d", delimiter=" ", encoding=None)

# =============================================================================
# Principal stress directions
# =============================================================================

ps1 = mesh.faces_attribute(name=tag, keys=mesh.faces())
ps1 = [normalize_vector(vector) for vector in ps1]

PS1 = np.array(ps1)
print("PS1 shape: ", PS1.shape)
print("PS1 first row: {}".format(PS1[0,:]))
print("PS1 last row: {}".format(PS1[-1,:]))

ps2 = mesh.faces_attribute(name=tag_2, keys=mesh.faces())
ps2 = [normalize_vector(vector) for vector in ps2]

PS2 = np.array(ps2)
print("PS2 shape: ", PS2.shape)
print("PS2 first row: {}".format(PS2[0,:]))
print("PS2 last row: {}".format(PS2[-1,:]))

np.savetxt(THERE + "ps1.txt", PS1, fmt="%1.6f", delimiter=" ", encoding=None)
np.savetxt(THERE + "ps2.txt", PS2, fmt="%1.6f", delimiter=" ", encoding=None)

print("Dot product first row PS1 - PS2: {}".format(np.dot(PS1[0, :], PS2[0,:].T)))
