# os
import os

# argument parsing helper
import fire

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

def results_plotting(filename,
                     plot_faces=True,
                     paint_clusters=True,
                     plot_mesh_edges=False,
                     plot_vector_fields=True,
                     plot_original_field=False,
                     plot_cones=False):
    """
    Makes a 3d plot of a mesh with a vector field.

    Parameters
    ----------
    filename : `str`
        The name of the JSON file that stores the clustering resultes w.r.t certain
        \n mesh, attribute, alglrithm and number of clusters.

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

        # supported vector field attributes
        available_vf = mesh_to_plot.vector_fields()
        print("Avaliable vector fields on the mesh are:\n", available_vf)

        # the name of the vector field to cluster.
        vf_names = []
        while True:
            vf_name = input("Please select the vector fields to cluster. Type 'ok' to stop adding vector fields: ")
            if vf_name in available_vf:
                vf_names.append(vf_name)
            elif vf_name == "ok":
                break
            else:
                print("This vector field is not available. Please try again.")

        for vf_name in vf_names:
            # clustered_field_name = vectorfield_tag + "_k"
            clustered_field_to_plot = mesh_to_plot.vector_field(vf_name)
            plotter.plot_vector_field_lines(mesh_to_plot, clustered_field_to_plot, (0, 0, 255), True, 0.07)

    # plot cones
    if plot_cones:
        vectors = mesh_to_plot.vector_field(vf_name)
        plotter.plot_vector_field_cones(mesh_to_plot, vectors)

    # set title, this will also set the final aspect ratio according to the data
    plotter.set_title(title="Example 01 Directional Clustering")

    #  show to screen
    plotter.show()


if __name__ == '__main__':
    fire.Fire(results_plotting)
