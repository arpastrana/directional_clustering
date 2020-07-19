from math import degrees
from math import fabs

from directional_clustering.geometry import clockwise
from directional_clustering.geometry import smoothed_angles

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

HERE = "../data/json_files/two_point_wall"

THERE = HERE.replace("json_files", "images")

base_vector_tag = "n_1"

transformable_vector_tags = ["n_1", "n_2"]
vector_cluster_tags = ["n_1_k", "n_2_k"]
vector_display_tags = ["n_2"]

smooth_iters = 0

vector_display_colors = [(0, 0, 255), (255, 0, 0)]  # blue and red
perp_flags = [False, True]
line_width = 1.0

ref_vector = [1.0, 0.0, 0.0]
mode = clockwise  # clockwise or cos_angle, angle calculation

n_clusters = 4  # "auto"  # int or auto
max_cl_iters = 20  # for auto mode
early_stopping = True
eps = 1.0

draw_kmeans_colors = False  # 2d representation

show_mesh = True
save_fig = False

data_to_color_tag = "clusters"  # angles, clusters, uncolored
draw_contours = False

draw_vector_field = False
vector_length = 0.005  # 0.005 if not uniform, 0.05 otherwise
uniform_length = False

export_json = False

# =============================================================================
# Import mesh
# =============================================================================

name = HERE.split("/").pop()
mesh = Mesh.from_json(HERE + ".json")
mesh_unify_cycles(mesh)

# =============================================================================
# Process PS vectors
# =============================================================================

angles = faces_angles(mesh, base_vector_tag, ref_vector, func=mode)

if smooth_iters:
    angles = smoothed_angles(mesh, angles, smooth_iters)

# ============================================================================
# Auto Kmeans angles
# =============================================================================

if n_clusters == "auto":
    shape = (-1, 1)
    a = angles
    errors = kmeans_errors(mesh, a, max_cl_iters, early_stopping, shape, eps)
    n_clusters = len(errors) + 1  # double check if + 1 later

# =============================================================================
# Kmeans angles
# =============================================================================

labels, centers = kmeans_clustering(angles, n_clusters, shape=(-1, 1))
print("centers", centers.shape)
for idx, c in enumerate(centers):
    print(idx, degrees(c), "deg")
print("labels", labels.shape)

# =============================================================================
# Quantized Colors
# =============================================================================

cluster_labels = faces_labels(mesh, labels, centers)

for fkey, label in cluster_labels.items():
    mesh.face_attribute(fkey, name="k_label", value=label)

# =============================================================================
# Register clustered field
# =============================================================================

for ref_tag, target_tag, perp in zip(transformable_vector_tags, vector_cluster_tags, perp_flags):
    faces_clustered_field(mesh, cluster_labels, ref_tag, target_tag, perp=perp, func=mode)

# =============================================================================
# data to plot
# =============================================================================

data_to_color = {
    "clusters": rgb_colors(cluster_labels),
    "angles": rgb_colors(angles),
    "uncolored": {}
    }

datacolors = data_to_color[data_to_color_tag]

# =============================================================================
# Kmeans plot 2d
# =============================================================================

if draw_kmeans_colors:
    plot_colored_vectors(centers, cluster_labels, angles, name)

# =============================================================================
# Set up Plotter
# =============================================================================

plotter = ClusterPlotter(mesh, figsize=(12, 9))
plotter.draw_edges(keys=list(mesh.edges_on_boundary()))
plotter.draw_faces(facecolor=datacolors)

# =============================================================================
# Scalar Contouring
# =============================================================================

if draw_contours:
    plotter.draw_clusters_contours(centers, cluster_labels, 100, "nearest")

# =============================================================================
# Create PS vector lines
# =============================================================================

if draw_vector_field:
    for tag, color in zip(vector_display_tags, vector_display_colors):
        plotter.draw_vector_field(tag, color, uniform_length, vector_length, line_width)

# =============================================================================
# Export json
# =============================================================================

if export_json:
    out = HERE + "_k_{}.json".format(n_clusters)
    mesh.to_json(out)
    print("Exported mesh to: {}".format(out))

if save_fig:
    out = THERE + "_field_{}.png".format(n_clusters)
    plotter.save(out, bbox_inches="tight")

# =============================================================================
# Show
# =============================================================================

if show_mesh:
    plotter.show()
