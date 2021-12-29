# os
import os

# argument parsing helper
import fire

# pyplot, 'cause why not
import matplotlib.pyplot as plt

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
                           manual_seeds=False,
                           comb_vectors=False,
                           align_vectors=False,
                           alignment_ref=[1.0, 0.0, 0.0],
                           smooth_iters=0,
                           smooth_align=True,
                           damping=0.5,
                           stress_transf_ref=[1.0, 0.0, 0.0],
                           output_align=True,
                           output_smooth_iters=0,
                           output_comb=False,
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
        vf_name = input("Please choose one vector field to cluster [m_1]: ")
        if vf_name in available_vf:
            break
        elif vf_name == "":
            vf_name = "m_1"
            break
        else:
            print("This vector field is not available. Please try again.")

    # ==========================================================================
    # Extract vector field from mesh for clustering
    # ==========================================================================

    vectors = mesh.vector_field(vf_name)

    # TODO: store a safety copy that won't undergo any transformations
    vectors_raw = mesh.vector_field(vf_name)

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
        print("-----")
        print(f"Aligning vector field to {alignment_ref}")
        align_vector_field(vectors, alignment_ref)

    # ==========================================================================
    # Comb the vector field -- remember the hair ball theorem (seams exist)
    # ==========================================================================

    if comb_vectors:
        print("-----")
        print("Combing vector field")
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
        print("-----")
        print(f"Smoothing vector field for {smooth_iters} iters. Align neighbors: {smooth_align}")
        smoothen_vector_field(vectors, mesh.face_adjacency(), smooth_iters, damping, smooth_align)

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

    # Create an instance of a clustering algorithm from ClusteringFactory
    clustering_algo = ClusteringFactory.create(algo_name)
    clusterer = clustering_algo(mesh, vectors)

    # Check if clustering method is differentiable
    # last word indicates diff
    is_clusterer_diff = algo_name.split("_")[-1] == "diff"

    # initialize seeds
    print("-----")
    if not manual_seeds:
        clusterer.seed(n_clusters, **kwargs_seeds)

    # TODO: manually setting seeds
    else:
        # manual_seeds = np.random.rand(n_clusters, 3)
        manual_seeds = np.array([[0.11417426, 0.28130369, 0.0],
                                 [0.60739818, 0.57142395, 0.0],
                                 [0.35134898, 0.85995537, 0.0],
                                 [0.45353768, 0.21977421, 0.0],
                                 [0.30653699, 0.50145481, 0.0]])

        clusterer.seeds = manual_seeds
        print("Manual seeds:\n", manual_seeds)

    # do kmeans clustering
    # labels contains the cluster index assigned to every vector in the vector field
    # centers contains the centroid of every cluster (the average of all vectors in
    # a cluster), and losses stores the losses generated per epoch.
    # the loss is simply the mean squared error of the cosine distance between
    # every vector and the centroid of the cluster it is assigned to
    # the goal of kmeans is to minimize this loss function

    print("-----")
    print("Clustering started...")
    clusterer.cluster(n_clusters, iters, tol, early_stopping, **kwargs)
    print(f"Loss Clustering (original field to {n_clusters} centroids): {clusterer.loss}")
    print("Clustering ended!")

    # store results in clustered_field and labels
    clustered_field = clusterer.clustered_field
    labels = clusterer.labels

    print("Computed centroids:\n")
    for index, center in clusterer.centers.items():
        print(f"{index}: {center}")

    # ==========================================================================
    # Print out gradient
    # ==========================================================================

    # if is_clusterer_diff:

        # from autograd import grad

        # recorder = {"attention": None,
        #             "centroids": None,
        #             "losses": [],
        #             "losses_field": []}
        # tau = 10.0
        # tau = np.array([1.0, 10.0, 20.0, 50.0, 100.0])  # smallest values lead to smaller gradients
        # tau = np.ones(n_clusters) * -1
        # stabilize = False
        # argnum = 5

        # grad_func = grad(clusterer._cluster_diff, argnum=argnum)

        # X = np.array(vectors.to_sequence())
        # seeds = clusterer.seeds

        # grad_cluster = grad_func(X, seeds, iters, tol, early_stopping, tau, stabilize, recorder)

        # # print(np.amax(np.abs(grad_cluster), axis=0)
        # print("-----")
        # print(f"Gradient w.r.t. argnum {argnum}:\n{np.abs(grad_cluster)}")

    # ==========================================================================
    # Compute mean squared error "loss" of clustering
    # ==========================================================================

    # TODO: probably would be better to encapsulate this in a function or in an object
    # clustering_error = MeanSquaredError(vector_field, clustered_field)
    # clustering_error = MeanAbsoluteError(vector_field, clustered_field)

    distances = np.zeros(mesh.number_of_faces())
    for fkey in mesh.faces():
        # for every face compute difference between clustering output and
        # aligned + smoothed vector, or versus raw input?
        distance = clusterer.distance_func(clustered_field.vector(fkey), vectors.vector(fkey))
        distances[fkey] = distance

    rmse = clusterer.loss_func(distances)
    print("-----")
    print(f"Clustered Field RMSE (clustered field to original field): {rmse}")

    # ==========================================================================
    # Plot clusterer loss history
    # ==========================================================================

    print("-----")
    while True:
        plot_history = input("Plot clusterer loss histories? [Y/n] ")
        if plot_history in ["", "y", "Y"]:
            plt.plot(clusterer.loss_history, label="loss_centroids")
            plt.plot(clusterer.loss_history_field, label="loss_field")
            plt.legend()
            plt.show()
            break
        elif plot_history == "n":
            break
        else:
            print("Hmm...That's neither a yes nor a no. Try again.")

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
    if is_clusterer_diff:
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
        # TODO: assumes mesh is planar so that vector to cross with is global Z
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
    stress_types = {"m": "bending", "n": "axial"}
    while True:
        stress_id = input("What stress type are we looking at, bending (m) or axial (n)? [M/n]")
        if stress_id in ["m", "n", ""]:
            if stress_id == "":
                stress_id = "m"
            stress_type = stress_types[stress_id]
            print(f"Selected stress type is: {stress_type}")
            break
        else:
            print("Hmm...That's neither axial (n) nor bending (m). Please try again.")

    args = [mesh, (clustered_field, clustered_field_90), stress_type, stress_transf_ref]
    clustered_field, clustered_field_90 = transformed_stress_vector_fields(*args)

    # ==========================================================================
    # Align clustered fields to input fields
    # ==========================================================================

    if output_align:
        # assumes vector fields pair (unmodified field, clustered+transformed field)
        for field, c_field in [(vectors_raw, clustered_field), (vectors_90, clustered_field_90)]:
            i = 0
            for fkey in mesh.faces():
                c_vector = c_field[fkey]
                vector = field[fkey]

                if dot_vectors(c_vector, vector) < 0.0:
                    c_vector = scale_vector(c_vector, -1.0)
                    c_field[fkey] = c_vector
                    i += 1

            print(f"Reversed {i}/{c_field.size()} vectors in clustered field before export!")

    # ==========================================================================
    # Comb the clustered vector fields -- remember the hair ball theorem
    # ==========================================================================

    if output_comb:
        print("-----")
        print("Combing both clustered fields 0 and 90")
        clustered_field = comb_vector_field(clustered_field, mesh)
        clustered_field_90 = comb_vector_field(clustered_field_90, mesh)

    # ==========================================================================
    # Apply smoothing to the clustered vector field
    # ==========================================================================

    if output_smooth_iters:
        print("-----")
        print(f"Smoothing output vector fields for {output_smooth_iters} iters. Align neighbors: {smooth_align}")
        smoothen_vector_field(clustered_field, mesh.face_adjacency(), output_smooth_iters, damping, smooth_align)
        smoothen_vector_field(clustered_field_90, mesh.face_adjacency(), output_smooth_iters, damping, smooth_align)

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
