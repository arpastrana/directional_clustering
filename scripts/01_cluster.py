# os
import os

# argument parsing helper
import fire

# good ol' numpy
import numpy as np

# geometry helpers
from compas.geometry import cross_vectors
from compas.geometry import length_vector
from compas.geometry import scale_vector
from compas.geometry import dot_vectors

# JSON file directory
from directional_clustering import JSON

# extended version of Mesh
from directional_clustering.mesh import MeshPlus

# clustering algorithms factory
from directional_clustering.clustering import ClusteringFactory
from directional_clustering.clustering import distance_cosine
from directional_clustering.clustering import distance_cosine_abs

# vector field
from directional_clustering.fields import VectorField

# transformations
from directional_clustering.transformations import align_vector_field
from directional_clustering.transformations import smoothen_vector_field
from directional_clustering.transformations import comb_vector_field
from directional_clustering.transformations import transformed_stress_vector_fields


# ==============================================================================
# Main function: directional_clustering
# ==============================================================================


def directional_clustering(filename,
                           algo_name="cosine_kmeans",
                           n_clusters=4,
                           iters=100,
                           tol=1e-6,
                           early_stopping=False,
                           comb_vectors=False,
                           align_vectors=False,
                           alignment_ref=[1.0, 0.0, 0.0],
                           smooth_iters=0,
                           damping=0.5,
                           stress_transf_ref=[1.0, 0.0, 0.0],
                           kwargs_seeds={},
                           kwargs={}):
    """
    Clusters a vector field that has been defined on a mesh. Exports a JSON file.

    Parameters
    ----------
    filename : `str`
        The name of the JSON file that encodes a mesh.
        All JSON files must reside in this repo's data/json folder.

    algo_name : `str`
        The name of the algorithm to cluster the vector field.
        \nSupported options are `cosine_kmeans` and `variational_kmeans`.

    n_clusters : `int`
        The number of clusters to generate.

    iters : `int`
        The number of iterations to run the clustering algorithm for.

    tol : `float`
        A small threshold value that marks clustering convergence.
        \nDefaults to 1e-6.

    align_vectors : `bool`
        Flag to align vectors relative to a reference vector.
        \nDefaults to False.

    alignment_ref : `list` of `float`
        The reference vector for alignment.
        \nDefaults to [1.0, 0.0, 0.0].

    smooth_iters : `int`
        The number iterations of Laplacian smoothing on the vector field.
        \nIf set to 0, no smoothing will take place.
        Defaults to 0.

    damping : `float`
        A value between 0.0 and 1.0 to control the intensity of the smoothing.
        \nZero technically means no smoothing. One means maximum smoothing.
        Defaults to 0.5.
    """

    # ==========================================================================
    # Set directory of input JSON files
    # ==========================================================================

    # Relative path to the JSON file stores the vector fields and the mesh info
    # The JSON files must be stored in the data/json_files folder

    name_in = filename + ".json"
    json_in = os.path.abspath(os.path.join(JSON, name_in))

    # ==========================================================================
    # Import a mesh as an instance of MeshPlus
    # ==========================================================================

    mesh = MeshPlus.from_json(json_in)

    # ==========================================================================
    # Search for supported vector field attributes and take one choice from user
    # ==========================================================================

    # supported vector field attributes
    available_vf = mesh.vector_fields()
    print("Avaliable vector fields on the mesh are:\n", available_vf)

    # the name of the vector field to cluster.
    while True:
        vf_name = input("Please choose one vector field to cluster:")
        if vf_name in available_vf:
            break
        else:
            print("This vector field is not available. Please try again.")

    # ==========================================================================
    # Extract vector field from mesh for clustering
    # ==========================================================================

    vectors = mesh.vector_field(vf_name)

    # ==========================================================================
    # Align vector field to a reference vector
    # ==========================================================================

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

    # ==========================================================================
    # Comb the vector field -- remember the hair ball theorem (seams exist)
    # ==========================================================================

    if comb_vectors:
        vectors = comb_vector_field(vectors, mesh)

    # ==========================================================================
    # Apply smoothing to the vector field
    # ==========================================================================

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

    # ==========================================================================
    # Do K-means Clustering
    # ==========================================================================

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
    # These seeds will be used later on as input to start the final kmeans run.

    print("-----")

    # Create an instance of a clustering algorithm from ClusteringFactory
    clustering_algo = ClusteringFactory.create(algo_name)
    clusterer = clustering_algo(mesh, vectors)

    # initialize seeds
    clusterer.seed(n_clusters, **kwargs_seeds)

    # do kmeans clustering
    # labels contains the cluster index assigned to every vector in the vector field
    # centers contains the centroid of every cluster (the average of all vectors in
    # a cluster), and losses stores the losses generated per epoch.
    # the loss is simply the mean squared error of the cosine distance between
    # every vector and the centroid of the cluster it is assigned to
    # the goal of kmeans is to minimize this loss function

    print("Clustering started...")
    clusterer.cluster(n_clusters, iters, tol, early_stopping, **kwargs)
    print(f"Loss Clustering: {clusterer.loss}")
    print("Clustering ended!")

    # store results in clustered_field and labels
    clustered_field = clusterer.clustered_field
    labels = clusterer.labels

    for index, center in clusterer.centers.items():
        print(f"{index}: {center}")

    # ==========================================================================
    # Compute mean squared error "loss" of clustering
    # ==========================================================================

    # TODO: probably would be better to encapsulate this in a function or in an object
    # clustering_error = MeanSquaredError(vector_field, clustered_field)
    # clustering_error = MeanAbsoluteError(vector_field, clustered_field)

    errors = np.zeros(mesh.number_of_faces())
    for fkey in mesh.faces():
        # for every face compute difference between clustering output and
        # aligned+smoothed vector, might be better to compare against the
        # raw vector
        # difference_vector = subtract_vectors(clustered_field.vector(fkey), vectors.vector(fkey))
        # errors[fkey] = length_vector_sqrd(difference_vector)
        error = clusterer.distance_func(clustered_field.vector(fkey), vectors.vector(fkey))
        errors[fkey] = np.square(error)

    rmse = np.sqrt(np.mean(errors))
    print("-----")
    print(f"Clustered Field RMSE: {rmse}")

    # ==========================================================================
    # Assign cluster labels to mesh
    # ==========================================================================

    mesh.cluster_labels("cluster", labels)

    # ==========================================================================
    # Store clusterer algorithm name as mesh attribute
    # ==========================================================================

    mesh.attributes["clusterer_name"] = algo_name

    # ==========================================================================
    # Assign attention coefficients to mesh - poor man's version
    # ==========================================================================

    # TO DO: refactor into a MeshPlus method
    is_diff = algo_name.split("_")[-1] == "diff"  # last word indicates diff
    if is_diff:
        mesh.attributes["attention"] = clusterer.attention
        tao = kwargs.get("tau")
        if tao:
            mesh.attributes["tau"] = tao

    # ==========================================================================
    # Generate field orthogonal to the clustered field
    # ==========================================================================

    # add perpendicular field tha preserves magnitude
    # assumes that vector field name has format foo_bar_1 or baz_2
    vf_name_parts = vf_name.split("_")
    for idx, entry in enumerate(vf_name_parts):
        # exits at the first entry
        if entry.isnumeric():
            dir_idx = idx
            direction = entry

    n_90 = 2
    if direction == n_90:
        n_90 = 1

    vf_name_parts[dir_idx] = str(n_90)
    vf_name_90 = "_".join(vf_name_parts)

    vectors_90 = mesh.vector_field(vf_name_90)
    clustered_field_90 = VectorField()

    for fkey, vector in clustered_field.items():
        cvec_90 = cross_vectors(clustered_field[fkey], [0, 0, 1])

        scale = length_vector(vectors_90[fkey])

        if dot_vectors(cvec_90, vectors_90[fkey]):
            scale *= -1.0

        cvec_90 = scale_vector(cvec_90, scale)
        clustered_field_90.add_vector(fkey, cvec_90)

    # ==========================================================================
    # Scale fields based on plane stress transformations
    # ==========================================================================

    print("-----")
    print("Rescaling vector field based on plane stress transformation")
    while True:
        stress_type = input("What stress type are we looking at, bending or axial? ")

        if stress_type in ["bending", "axial"]:
            break
        else:
            print("Hmm...That's neither axial nor bending. Please try again.")

    args = [mesh, (clustered_field, clustered_field_90), stress_type, stress_transf_ref]
    clustered_field, clustered_field_90 = transformed_stress_vector_fields(*args)

    # ==========================================================================
    # Assign clustered fields to mesh
    # ==========================================================================

    clustered_field_name = vf_name + "_k"
    mesh.vector_field(clustered_field_name, clustered_field)

    clustered_field_name_90 = vf_name_90 + "_k"
    mesh.vector_field(clustered_field_name_90, clustered_field_90)

    # ==========================================================================
    # Export new JSON file for further processing
    # ==========================================================================

    def string_collapser(string, character="_"):
        split_string = string.split(character)
        return "".join(split_string)

    name_clustering = (filename, f"k_{n_clusters}", vf_name, algo_name)
    name_clustering = "_".join([string_collapser(s) for s in name_clustering])
    name_transforms = f"align{int(align_vectors)}_comb{int(comb_vectors)}_smooth{smooth_iters}"
    name_out = "_".join([name_clustering, name_transforms]) + ".json"

    json_out = os.path.abspath(os.path.join(JSON, "clustered", algo_name, name_out))
    mesh.to_json(json_out)

    print("-----")
    print(f"Exported clustered vector field with mesh to:\n{json_out}")

# ==============================================================================
# Main
# ==============================================================================


if __name__ == '__main__':
    fire.Fire(directional_clustering)
