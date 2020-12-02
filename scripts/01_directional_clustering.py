# os
import os

# good ol' numpy
import numpy as np

# this are ready-made functions from COMPAS (https://compas.dev)
from compas.datastructures import Mesh

# geometry helpers
from compas.geometry import subtract_vectors
from compas.geometry import length_vector_sqrd

# clustering algorithms factory
from directional_clustering.clustering import ClusteringFactory

# vector field
from directional_clustering.fields import VectorField

# transformations
from directional_clustering.transformations import align_vector_field
from directional_clustering.transformations import smoothen_vector_field

# these are custom-written functions part of this library
# which you can find in the src/directional_clustering folder
from directional_clustering import JSON
from directional_clustering.plotters import ClusterPlotter
from directional_clustering.plotters import rgb_colors

# =============================================================================
# Available Vector Fields
# =============================================================================

# Just for reference, this is a list with all available vector fields
# which have been stored in a JSON file.
#
# The JSON file will be loaded to create a COMPAS mesh.
#
# A vector field here is just a cloud of vectors, where there always is a
# single vector associated with the centroid of every face of the mesh.
# E.g., if a mesh has N faces, there will be a vector field with N vectors.
#
# First and second principal vectors are always orthogonal to each other.

vectorfield_tags = [
    "n_1",  # axial forces in first principal direction
    "n_2",  # axial forces in second principal direction
    "m_1",  # bending moments in first principal direction
    "m_2",  # bending moments in second principal direction
    "ps_1_top",  # first principal direction stress direction at topmost fiber
    "ps_1_bot",  # first principal direction stress direction at bottommost fiber
    "ps_1_mid",  # first principal direction stress direction at middle fiber
    "ps_2_top",  # second principal direction stress direction at topmost fiber
    "ps_2_bot",  # second principal direction stress direction at bottommost fiber
    "ps_2_mid",  # second principal direction stress direction at middle fiber
    "custom_1",  # an arbitrary vector field pointing in the global X direction
    "custom_2"   # an arbitrary vector field pointing in the global X direction
    ]

# ==============================================================================
# Set pipeline parameters
# ==============================================================================

# Relative path to the JSON file stores the vector fields and the mesh info
# The JSON files are stored in the data/json_files folder
# I am working on MacOS, so the format might be slightly different on Windows
name_in = "perimeter_supported_slab.json"
name_out = "perimeter_supported_slab_k_5.json"

JSON_IN = os.path.abspath(os.path.join(JSON, name_in))
JSON_OUT = os.path.abspath(os.path.join(JSON, name_out))

# vector field
vectorfield_tag = "m_1"  # the vector field to base the clustering on

# reference vector for alignment
align_vectors = True
alignment_ref = [1.0, 0.0, 0.0]  # global x

# smoothing
smooth_iters = 10  # how many iterations to run the smoothing for
damping = 0.5  # damping coefficient, a value from 0 to 1

# kmeans clustering
clustering_name = "variational kmeans" # algorithm name
n_clusters = 5  # number of clusters to produce
tol = 1e-6  # loss function threshold for early stopping
iters = 30 # number of epochs to run kmeans clustering for

# exporting
export_json = False

# plotter flags
draw_faces = True
draw_vector_fields = False

# ==============================================================================
# Import a COMPAS mesh
# ==============================================================================

mesh = Mesh.from_json(JSON_IN)

# ==============================================================================
# Extract vector field from COMPAS mesh for clustering
# ==============================================================================

# Extract a vector field from the faces of a mesh
vectors = VectorField.from_mesh_faces(mesh, vectorfield_tag)

# ==============================================================================
# Align vector field to a reference vector
# ==============================================================================

# the output of the FEA creates vector fields that are oddly oriented.
# Eventually, what we want is to create "lines" from this vector
# field that can be materialized into reinforcement bars, beams, or pipes which
# do not really care about where the vectors are pointing to.
# concretely, a vector can be pointing to [1, 1] or to [-1, 1] but for archi-
# tectural and structural reasons this would be the same, because both versions
# are colinear.
#
# in short, mitigating directional duplicity is something we are kind of
# sorting out with a heuristic. this will eventually improve the quality of the
# clustering
#
# how to pick the reference vector is arbitrary ("user-defined") and something
# where there's more work to be done on. in the meantime, i've used the global
# x and global y vectors as references, which have worked ok for my purposes.

if align_vectors:
    align_vector_field(vectors, alignment_ref)

# ==============================================================================
# Apply smoothing to the vector field
# ==============================================================================

# moreover, depending on the quality of the initial mesh, the FEA-produced
# vector field will be very noisy, especially around "singularities".
# this means there can be drastic orientation jumps/flips between two vectors
# which will affect the quality of the clustering.
# to mitigate this, we apply laplacian smoothing which helps to soften and
# preserve continuity between adjacent vectors
# what this basically does is going through every face in the mesh,
# querying what are their neighbor faces, and then averaging the vectors stored
# on each of them to finally average them all together. the "intensity" of this
# operation is controlled with the number of smoothing iterations and the
# damping coefficient.
# too much smoothing however, will actually distort the initial field to the
# point that is not longer "representing" the original vector field
# so smoothing is something to use with care

if smooth_iters:
    smoothen_vector_field(vectors, mesh.face_adjacency(), smooth_iters, damping)

# ==============================================================================
# Do K-means Clustering
# ==============================================================================

# now we need to put the vector field into numpy arrays to carry out clustering
# current limitation: at the moment this only works in planar 2d meshes!
# other clustering methods, like variational clustering, can help working
# directly on 3d by leveraging the mesh datastructure
# other ideas relate to actually "reparametrizing" (squishing) a 3d mesh into a
# 2d mesh, carrying out clustering directly in 2d, and then reconstructing
# the results back into the 3d mesh ("reparametrizing it back")

# One of the key differences of this work is that use cosine distance as
# basis metric for clustering. this is in constrast to numpy/scipy
# whose implementations, as far as I remember support other types of distances
# like euclidean or manhattan in their Kmeans implementations. this would not
# work for the vector fields used here, but maybe this has changed now.
# in any case, this "limitation" led me to write our own version of kmeans
# which can do clustering based either on cosine or euclidean similarity

# kmeans is sensitive to initialization
# there are a bunch of approaches that go around it, like kmeans++
# i use here another heuristic which iteratively to find the initial seeds
# using a furthest point strategy, which basically picks as a new seed the
# vector which is the "most distant" at a given iteration using kmeans itself
#
# These seeds will be used later on as input to start the final kmeans run.

print("Clustering started...")

# Create an instance of a clustering algorithm
clustering_algo = ClusteringFactory.create(clustering_name)
clusterer = clustering_algo(mesh, vectors, n_clusters, iters, tol)

# do kmeans clustering
# what this method returns are three numpy arrays
# labels contains the cluster index assigned to every vector in the vector field
# centers contains the centroid of every cluster (the average of all vectors in
# a cluster), and losses stores the losses generated per epoch.
# the loss is simply the mean squared error of the cosine distance between
# every vector and the centroid of the cluster it is assigned to
# the goal of kmeans is to minimize this loss function

clusterer.cluster()

print("Loss Clustering: {}".format(clusterer.loss))
print("Clustering ended!")

# make an array with the assigned labels of all vectors
clustered_field = clusterer.clustered_field
labels = clusterer.labels

# ==============================================================================
# Compute mean squared error "loss" of clustering w.r.t. original vector field
# ==============================================================================

# probably would be better to encapsulate this in a function or in a Loss object
# clustering_error = MeanSquaredError(vector_field, clustered_field)
# clustering_error = MeanAbsoluteError(vector_field, clustered_field)

errors = np.zeros(mesh.number_of_faces())
for fkey in mesh.faces():
    # for every face compute difference between clustering output and
    # aligned+smoothed vector, might be better to compare against the
    # raw vector
    difference_vector = subtract_vectors(clustered_field.vector(fkey),
                                         vectors.vector(fkey))
    errors[fkey] = length_vector_sqrd(difference_vector)

mse = np.mean(errors)

print("Clustered Field MSE: {}".format(mse))

# ==============================================================================
# Assign clusters back to COMPAS mesh
# ==============================================================================

# this is for visualization and exporting purposes
attr_name = vectorfield_tag + "_k_{}".format(n_clusters)  # name for storage

# iterate over the faces of the COMPAS mesh
for fkey in mesh.faces():
    c_vector = clustered_field.vector(fkey)  # convert numpy array to list
    c_label = labels[fkey]

    # store clustered vector in COMPAS mesh as a face attribute
    mesh.face_attribute(key=fkey, name=attr_name, value=c_vector)
    mesh.face_attribute(key=fkey, name="cluster", value=c_label)

# ==============================================================================
# Export new JSON file for further processing
# ==============================================================================

if export_json:
    mesh.to_json(JSON_OUT)
    print("Exported mesh to: {}".format(JSON_OUT))

# =============================================================================
# Plot stuff
# =============================================================================

# convert vectors dictionary into a numpy array
vectors_array = np.zeros((mesh.number_of_faces(), 3))
for fkey, vec in vectors.items():
    vectors_array[fkey, :] = vec

# there is a lot of potential work to do for visualization!
# below there is the simplest snippet, but you can see more stuff
# in the scripts/visualization folder

# ClusterPlotter is a custom wrapper around a COMPAS MeshPlotter
# the COMPAS MeshPlotter is built atop of pure Matplotlib (which is crazy)
# what is different here is that I extended the plotter so that it can plot
# vector fields directly as little lines via
# ClusterPlotter.draw_vector_field_array()
plotter = ClusterPlotter(mesh, figsize=(12, 9))

# draw only the boundary edges of the COMPAS Mesh
plotter.draw_edges(keys=list(mesh.edges_on_boundary()))

if draw_faces:
    # color up the faces of the COMPAS mesh according to their cluster
    # make a dictionary with all labels
    labels_to_color = {}
    for fkey in mesh.faces():
        labels_to_color[fkey] = mesh.face_attribute(key=fkey, name="cluster")
    # convert labels to rgb colors
    face_colors = rgb_colors(labels_to_color, invert=False)
    # draw faces
    plotter.draw_faces(facecolor=face_colors)

# draw vector fields on mesh as lines
if draw_vector_fields:
    # original vector field
    va = vectors_array  # shorthand
    plotter.draw_vector_field_array(va, (50, 50, 50), True, 0.07, 0.5)
    # clustered vector field
    # plotter.draw_vector_field_array(clusters, (0, 0, 255), True, 0.07, 1.0)

#  show to screen
plotter.show()
