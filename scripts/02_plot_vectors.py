# os
import os

# good ol' numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt

# this are ready-made functions from COMPAS (https://compas.dev)
from compas.datastructures import Mesh

from compas.geometry import scale_vector
from compas.geometry import dot_vectors
from compas.geometry import subtract_vectors
from compas.geometry import length_vector_sqrd
from compas.geometry import normalize_vector
from compas.geometry import length_vector

# this are custom-written functions part of this library
# which you can find in the src/directional_clustering folder
from directional_clustering import JSON
from directional_clustering.geometry import laplacian_smoothed
from directional_clustering.geometry import cosine_similarity
from directional_clustering.clusters import init_kmeans_farthest
from directional_clustering.clusters import kmeans
from directional_clustering.plotters import ClusterPlotter
from directional_clustering.plotters import rgb_colors
from directional_clustering.plotters import plot_kmeans_vectors

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

vectorfield_tags= [
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
JSON_IN = os.path.abspath(os.path.join(JSON, name_in))

# vector field
vectorfield_tag = "m_1"  # the vector field to base the clustering on

# reference vector for alignment
align_vectors = True
alignment_ref = [1.0, 0.0, 0.0]  # global x
# alignment_ref = [0.0, 1.0, 0.0]  # global y

# smoothing
smooth_iters = 0  # how many iterations to run the smoothing for
damping = 0.5  # damping coefficient, a value from 0 to 1

# kmeans clustering
n_clusters = 3  # number of clusters to produce
mode = "cosine"  # "cosine" or "euclidean"
eps = 1e-6  # loss function threshold for early stopping
epochs_seeds = 100  # number of epochs to run the farthest seeding for
epochs_kmeans = 100  # number of epochs to run kmeans clustering for

# reference vector for alignment
cosim_ref = [0.0, 1.0, 0.0]  # global y


# plot vectors in 2d
plot_vectors_2d = True
normalize = False
rescale = False
vec_scale = 1.0  # for rescaling, max length

# plot cosine similarity as height field
plot_cosine_similarity_3d = True

# plot clustered vectors in 2d
plot_clustered_vectors = True

# ==============================================================================
# Import a COMPAS mesh
# ==============================================================================

mesh = Mesh.from_json(JSON_IN)

# ==============================================================================
# Extract vector field from COMPAS mesh for clustering
# ==============================================================================

# first, create a dict mapping from face keys to indices to remember what
# vector belonged to what face. This will be handy when we plot the clustering
# results
fkey_idx = {fkey: index for index, fkey in enumerate(mesh.faces())}

# store vector field in a dictionary where keys are the mesh face keys
# and the values are the vectors located at every face
vectors = {}
for fkey in mesh.faces():
    # this is a mesh method that will query info stored the faces of the mesh
    vectors[fkey] = mesh.face_attribute(fkey, vectorfield_tag) 

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
    for fkey, vector in vectors.items():    
        # if vectors don't point in the same direction
        if dot_vectors(alignment_ref, vector) < 0.0:
            vectors[fkey] = scale_vector(vector, -1)  # reverse it

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
    vectors = laplacian_smoothed(mesh, vectors, smooth_iters, damping)

# ==============================================================================
# Do K-means Clustering ================================================================================

# functions related to kmeans are in src/directional_clustering/clusters/

# now we need to put the vector field into numpy arrays to carry out clustering
# current limitation: at the moment this only works in planar 2d meshes!
# other clustering methods, like variational clustering, can help working
# directly on 3d by leveraging the mesh datastructure
# other ideas relate to actually "reparametrizing" (squishing) a 3d mesh into a
# 2d mesh, carrying out clustering directly in 2d, and then reconstructing
# the results back into the 3d mesh ("reparametrizing it back")

# convert vectors dictionary into a numpy array 
vectors_array = np.zeros((mesh.number_of_faces(), 3))
for fkey, vec in vectors.items():
    vectors_array[fkey, :] = vec

print("Clustering started...")

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

seeds = init_kmeans_farthest(vectors_array, n_clusters, mode, epochs_seeds, eps)

# do kmeans clustering
# what this method returns are three numpy arrays
# labels contains the cluster index assigned to every vector in the vector field
# centers contains the centroid of every cluster (the average of all vectors in
# a cluster), and losses stores the losses generated per epoch. 
# the loss is simply the mean squared error of the cosine distance between
# every vector and the centroid of the cluster it is assigned to
# the goal of kmeans is to minimize this loss function

labels, centers, losses = kmeans(vectors_array, seeds, mode, epochs_kmeans, eps, early_stopping=True, verbose=True)

print("loss kmeans", losses[-1])
print("Clustering ended!")

# make an array with the assigned labels of all vectors
clusters = centers[labels]

# ==============================================================================
# Plot vectors 2d
# ==============================================================================

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
# Cosine similarity
# =============================================================================

cosim = np.zeros(mesh.number_of_faces())
for fkey, vec in vectors.items():
    cosim[fkey] = cosine_similarity(cosim_ref, vec) 

# =============================================================================
# Plot cosine similarity as heights
# =============================================================================

if plot_cosine_similarity_3d:

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
# Plot clustered vectors in 2d
# =============================================================================

if plot_clustered_vectors:
    plot_kmeans_vectors(vectors_array, labels, centers, normalize=False, draw_centroids=True)
