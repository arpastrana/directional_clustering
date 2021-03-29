# os
import os

# argument parsing helper
import fire

# these are custom-written functions part of this library
# which you can find in the src/directional_clustering folder

#JSON file directory
from directional_clustering import JSON

# extended version of Mesh
from directional_clustering.mesh import MeshPlus

# clustering algorithms factory
from directional_clustering.clustering import ClusteringFactory

# vector field
from directional_clustering.fields import VectorField

#plotters
from directional_clustering.plotters import PlyPlotter


# ==============================================================================
# Main function: directional_clustering
# ==============================================================================

def results_plotting(filename="perimeter_supported_slab_m_1_variational kmeans_5",
                     vectorfield_tag="m_1",
                     plot_faces=True,
                     paint_clusters=True,
                     plot_mesh_edges=False,
                     plot_vector_fields=True,
                     plot_original_field=False,
                     plot_cones=False):
    """
    Plot clustering results stored in JSON file.

    Parameters
    ----------
    filename : `str`
        The name of the JSON file that stores the clustering resultes w.r.t certain
        \n mesh, attribute, alglrithm and number of clusters.
        \nAll JSON files must reside in this repo's data/json folder.
        Defaults to "perimeter_supported_slab_m_1_variational kmeans_5".

    vectorfield_tag : `str`
        The name of the vector field on which clustering has been done, should corresponds
        to the `filename`.
        Defaults to "m_1"

    plot_faces : `bool`
        Plots the faces of the input mesh.
        \nDefaults to True.

    paint_clusters : `bool`
        Color up the faces according to their cluster
        \nDefaults to True.

    plot_mesh_edges : `bool`
        Plots the edges of the input mesh.
        \nDefaults to False.

    plot_vector_fields : `bool`
        Plots the clustered vector field atop of the input mesh.
        \nDefaults to True.

    plot_original_field : `bool`
        Plots the original vector field before clustering atop of the input mesh.
        \nDefaults to False.

    plot_cones : `bool`
        Plots the cones atop of the input mesh.
        \nDefaults to False.
   """

    # ============================================================================
    # Plot stuff
    # ============================================================================

    # there is a lot of potential work to do for visualization!
    # below there is the simplest snippet, but you can see more stuff
    # in the scripts/visualization folder

    #resume results from JSON file
    name_in = filename + ".json"
    json_in = os.path.abspath(os.path.join(JSON, name_in))

    mesh_to_plot = MeshPlus.from_json(json_in)

    # there is a lot of potential work to do for visualization!
    # below there is the simplest snippet, but you can see more stuff
    # in the scripts/visualization folder

    # PlyPlotter is a custom wrapper around a Plotly graph object (Figure)
    # that handles formating and adjustments to data structure.
    plotter = PlyPlotter()

    # color up the faces of the mesh according to their cluster
    if plot_faces:
        plotter.plot_trimesh(mesh_to_plot, paint_clusters, plot_mesh_edges)

    # plot vector fields on mesh as lines
    if plot_vector_fields:
        clustered_field_name = vectorfield_tag + "_clustered"
        clustered_field_to_plot = mesh_to_plot.vector_field(
            clustered_field_name)
        plotter.plot_vector_field_lines(mesh_to_plot, clustered_field_to_plot,
            (0, 0, 255), True, 0.07)
    if plot_original_field:
        vectors = mesh_to_plot.vector_field(vectorfield_tag)
        plotter.plot_vector_field_lines(mesh_to_plot, vectors, (50, 50, 50),
            True, 0.07)

    # plot cones
    if plot_cones:
        vectors = mesh_to_plot.vector_field(vectorfield_tag)
        plotter.plot_vector_field_cones(mesh_to_plot, vectors)

    # set title, this will also set the final aspect ratio according to the data
    plotter.set_title(title="Example 01 Directional Clustering")

    #  show to screen
    plotter.show()


if __name__ == '__main__':
    fire.Fire(results_plotting)
