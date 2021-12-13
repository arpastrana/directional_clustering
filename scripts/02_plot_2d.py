# os
import os

# argument parsing helper
import fire

# hello numpy, my old friend
import numpy as np

# plots and beyond
import matplotlib.pyplot as plt

# python standard libraries
from itertools import cycle
from functools import partial

# time is running out
from datetime import datetime

# compas & co
from compas.geometry import angle_vectors
from compas.geometry import length_vector
from compas.geometry import KDTree
from compas.utilities import geometric_key_xy

# Library file directories
from directional_clustering import DATA
from directional_clustering import JSON

# extended version of Mesh
from directional_clustering.mesh import MeshPlus

# clustering
from directional_clustering.clustering import distance_cosine
from directional_clustering.clustering import distance_cosine_abs

# transformations
from directional_clustering.transformations import align_vector_field
from directional_clustering.transformations import comb_vector_field

# plotters
from directional_clustering.plotters import MeshPlusPlotter


# ==============================================================================
# Matplotlib beautification
# ==============================================================================

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('axes', linewidth=1.5)
plt.rc('axes', labelsize=15)
plt.rc('xtick', labelsize=20, direction="in")
plt.rc('ytick', labelsize=20, direction="in")
plt.rc('legend', fontsize=15)

# setting xtick parameters
plt.rc('xtick.major', size=10, pad=4)
plt.rc('xtick.minor', size=5, pad=4)
plt.rc('ytick.major', size=10)
plt.rc('ytick.minor', size=5)

# ==============================================================================
# Set global variables
# ==============================================================================

ZORDER_LEGEND = 5000  # manually set to be drawn in front of other objects
IMG_EXTENSION = "pdf"
CMAP_LABELS = "rainbow"

# Set the colors to use to color mesh faces/centroids
COLOR_LABEL_MAP = {"angles": {"cmap": "RdPu",
                              "cbar_label": "Angular Difference [Deg]",
                              "func": partial(angle_vectors, deg=True)},
                   "cosine_distance": {"cmap": "RdPu",
                                       "cbar_label": "Cosine Distance",
                                       "func": distance_cosine},
                   "abs_cosine_distance": {"cmap": "RdPu",
                                           "cbar_label": "Absolute Cosine Distance",
                                           "func": distance_cosine_abs}
                   }

# ==============================================================================
# Plot a lot of information in 2d
# ==============================================================================


def plot_2d(filename,
            draw_faces=True,
            draw_faces_centroids=False,
            draw_vector_fields=False,
            draw_streamlines=False,
            draw_edges=False,
            draw_colorbar=True,
            draw_boundary_edges=True,
            color_faces=None,
            comb_fields=False,
            align_field_1_to=None,
            align_field_2_to=None,
            streamlines_density=0.75,  # 0.55 for 4ps
            streamlines_lw=None,
            vector_fields_scale=0.03,
            vector_fields_same_scale=True,
            save_img=True,
            pad_inches=0.0,
            show_img=False
            ):
    """
    Makes a 3d plot of a mesh with a vector field.

    Parameters
    ----------
    filename : `str`
        The name of the JSON file that stores the clustering resultes w.r.t certain
        \n mesh, attribute, alglrithm and number of clusters.
    """
    # ==========================================================================
    # Load up a mesh from a JSON file
    # ==========================================================================

    name_in = filename + ".json"
    json_in = os.path.abspath(os.path.join(JSON, "clustered", name_in))

    mesh = MeshPlus.from_json(json_in)

    # ==========================================================================
    # Instantiate a plotter
    # ==========================================================================

    # ClusterPlotter is a custom wrapper around a COMPAS MeshPlotter
    plotter = MeshPlusPlotter(mesh, figsize=(16, 9), dpi=600)

    # ==========================================================================
    # Draw mesh edges
    # ==========================================================================

    if draw_boundary_edges:
        plotter.draw_edges(keys=list(mesh.edges_on_boundary()))

    # draw mesh edges
    if draw_edges:
        plotter.draw_edges()

    # ==========================================================================
    # Draw mesh faces
    # ==========================================================================

    # get number of clusters from filename
    mesh_name = filename.split("/")[-1]
    n_clusters = int(mesh_name.split("_")[1][-1]) # second position is number of cluster
    print(f"The number of clusters on the mesh is: {n_clusters}")

    if draw_faces or draw_faces_centroids:
        cmap = None
        data = np.zeros(mesh.number_of_faces())
        sorted_fkeys = sorted(list(mesh.faces()))

        if color_faces == "labels":
            # parse cluster labels
            labels = mesh.cluster_labels("cluster")
            for fkey, label in labels.items():
                data[fkey] = label

            # matplotlib setup
            cbar_label = "Directional Clusters"
            cmap = plt.cm.get_cmap(CMAP_LABELS, n_clusters)  # plasma or rainbow
            ticks = np.linspace(0, n_clusters - 1, n_clusters + 1) + 0.5 * (n_clusters - 1)/n_clusters
            ticks = ticks[:-1]
            ticks_labels = list(range(n_clusters))
            extend = "neither"

        elif color_faces in {"angles", "cosine_distance", "abs_cosine_distance"}:

            available_vf = mesh.vector_fields()
            print("Avaliable vector fields on the mesh are:\n", available_vf)

            vf_names = []
            while len(vf_names) < 2:
                vf_name = input("Please select the two vector fields to compare: ")
                if vf_name in available_vf:
                    if vf_name not in vf_names:
                        vf_names.append(vf_name)
                else:
                    print("This vector field is not available. Please try again.")

            vf_name_a, vf_name_b = vf_names
            vf_a = mesh.vector_field(vf_name_a)
            vf_b = mesh.vector_field(vf_name_b)

            clm = COLOR_LABEL_MAP[color_faces]
            func = clm["func"]
            cmap = clm["cmap"]
            cbar_label = clm["cbar_label"]

            for fkey in mesh.faces():
                data[fkey] = func(vf_a[fkey], vf_b[fkey])

            ticks = np.linspace(data.min(), data.max(), num=7)
            ticks_labels = [np.round(x, 2) for x in ticks]
            extend = "both"

        elif color_faces == "attention":

            attention = mesh.attributes["attention"]

            while True:
                att_k = input(f"Please select the cluster attention coefficient to plot (0 to {n_clusters-1}): ")
                try:
                    att_k = int(att_k)
                    if att_k >= 0 and att_k < (n_clusters):
                        break
                except:
                    print("That cluster index doesn't exist. Try again.")

            # parse cluster labels
            for fkey, attention_coefficients in attention.items():
                # TODO: data serialization convert int-type fkeys into strings!
                data[int(fkey)] = attention_coefficients[att_k]

            # rename color_faces for export image name
            color_faces = color_faces + str(att_k)

            # matplotlib setup
            cbar_label = f"Attention Coefficient for Cluster {att_k}"
            cmap = "RdPu"
            ticks = np.linspace(data.min(), data.max(), num=7)
            ticks_labels = [np.round(x, 2) for x in ticks]
            extend = "both"

        elif color_faces == "labels_attention":

            attention = mesh.attributes["attention"]

            cluster_indices = np.array(list(range(0, n_clusters)))
            A = np.array(list([coeff for _, coeff in attention.items()]))
            cluster_attention = A @ cluster_indices

            for idx, bundle in enumerate(attention.items()):
                # TODO: data serialization convert int-type fkeys into strings!
                fkey, _ = bundle
                data[int(fkey)] = cluster_attention[idx]

            tao = mesh.attributes["tau"]
            cbar_label = r"Attention per Cluster, $\tau={}$".format(tao)
            cmap = CMAP_LABELS
            ticks = np.linspace(0, n_clusters-1, n_clusters)
            ticks_labels = list(range(n_clusters))
            extend = "neither"

        if draw_faces:

            collection = plotter.draw_faces(keys=sorted_fkeys)

        elif draw_faces_centroids:

            points = []
            for fkey in sorted_fkeys:
                point = {}
                point["pos"] = mesh.face_centroid(fkey)
                point["radius"] = 0.03
                point["edgewidth"] = 0.10
                points.append(point)

            collection = plotter.draw_points(points)

        if cmap:
            collection.set(array=data, cmap=cmap)
            collection.set_linewidth(lw=0.0)

            if draw_colorbar:
                colorbar = plt.colorbar(collection,
                                        shrink=0.9,
                                        pad=0.005,
                                        extend=extend,
                                        extendfrac=0.05,
                                        ax=plotter.axes,
                                        ticks=ticks,
                                        aspect=30,
                                        orientation="vertical")

                # colorbar.set_ticks(ticks)
                assert len(ticks) == len(ticks_labels)
                colorbar.ax.set_yticklabels(ticks_labels)
                colorbar.set_label(cbar_label, fontsize="large")

    # ==========================================================================
    # Draw vector field information (as lines or as streamlines)
    # ==========================================================================

    if draw_vector_fields or draw_streamlines:

        # supported vector field attributes
        available_vf = mesh.vector_fields()
        print("Avaliable vector fields on the mesh are:\n", available_vf)

        # the name of the vector field to cluster.
        vf_names = []

        while True:
            vf_name = input("Please select the vector fields to draw. Type 'ok' to stop adding vector fields: ")
            if vf_name in available_vf:
                if vf_name not in vf_names:
                    vf_names.append(vf_name)
            elif vf_name == "ok":
                break
            else:
                print("This vector field is not available. Please try again.")

        if draw_vector_fields:
            # colors for all vector related matters
            colors = cycle([(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 0)])

            # vector field drawing parameters -- better of being exposed?
            width = 0.5
            scale = vector_fields_scale
            same_scale = vector_fields_same_scale

            for vf_name in vf_names:
                vf = mesh.vector_field(vf_name)

                if comb_fields:
                    vf = comb_vector_field(vf, mesh)

                color = next(colors)
                lines = plotter.draw_vector_field(vf, color, same_scale, scale, width)
                _vf_name = "".join(vf_name.split("_"))
                lines.set_label(_vf_name)

        if draw_streamlines:
            colors = cycle([(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 0)])

            # get bounding box for all the mesh vertices
            vx = []
            vy = []
            for vkey in mesh.vertices():
                vx.append(mesh.vertex_attribute(vkey, name="x"))
                vy.append(mesh.vertex_attribute(vkey, name="y"))

            vxmin = min(vx)
            vxmax = max(vx)
            vymin = min(vy)
            vymax = max(vy)

            # create linear spaces on x and y
            # offset inwards by 1%
            pe = 0.01
            vxdiff = vxmax - vxmin
            vydiff = vymax - vymin

            X = np.linspace(vxmin + pe * vxdiff, vxmax - pe * vxdiff, len(set(vx)))
            Y = np.linspace(vymin + pe * vydiff, vymax - pe * vydiff, len(set(vy)))
            XX, YY = np.meshgrid(X, Y)

            # gather gkey maps
            gkey_fkey = {}
            gkey_xyz = {}
            xyz = []
            for fkey in mesh.faces():
                xyz.append(mesh.face_centroid(fkey))
                gkey = geometric_key_xy(mesh.face_centroid(fkey))
                gkey_xyz[gkey] = mesh.face_centroid(fkey)
                gkey_fkey[gkey] = fkey

            # build search tree
            search_tree = KDTree(objects=xyz)

            # select vectors for dot product alignment
            alignment_vectors = []
            alignment_map_xy = {"X": [1.0, 0.0, 0.0], "Y": [0.0, 1.0, 0.0]}
            for align_to in (align_field_1_to, align_field_2_to):
                if align_to in alignment_map_xy:
                    alignment_vectors.append(alignment_map_xy[align_to])

            if len(alignment_vectors) > 0:
                alignment_vectors_cycle = cycle(alignment_vectors)

            # do for every vector field
            for vf_name in vf_names:

                vf = mesh.vector_field(vf_name)

                # comb the line field prior to streamlines tracing
                if comb_fields:
                    vf = comb_vector_field(vf, mesh)

                if len(alignment_vectors) > 0:
                    alignment_vector = next(alignment_vectors_cycle)
                    align_vector_field(vf, alignment_vector)

                # query vectors from vector field
                U = []
                V = []
                LV = []

                if streamlines_lw is not None:
                    streamlines_lw = float(streamlines_lw)

                for xx, yy in zip(XX.flatten(), YY.flatten()):
                    # strictly 2d
                    test_xyz = [xx, yy, 0.0]
                    near_xyz, _, _ = search_tree.nearest_neighbor(test_xyz)

                    # generate grid gkey
                    gkey = geometric_key_xy(near_xyz)
                    fkey = gkey_fkey[gkey]
                    u, v = vf[fkey][:2]

                    U.append(u)
                    V.append(v)

                    # compute lineweights
                    LV.append(length_vector(vf[fkey]))

                U = np.reshape(U, XX.shape)
                V = np.reshape(V, XX.shape)


                LW = []
                if streamlines_lw:

                    for value in LV:
                        centered_val = (value - min(LV))  # in case min is not 0
                        ratio = centered_val / max(LV)
                        ratio = streamlines_lw * ratio
                        LW.append(ratio + 1)  # add one to not have invisible lines

                    LW = np.reshape(LW, XX.shape)

                # filter lineweights for match streamplot's signature
                if len(LW) == 0:
                    LW = None

                # plot streamlines
                stream_set = plt.streamplot(XX, YY, U, V,
                                            color=[i / 255.0 for i in next(colors)],
                                            arrowsize=0.0,
                                            maxlength=20.0,
                                            minlength=0.1,
                                            density=streamlines_density,
                                            integration_direction="both",
                                            linewidth=LW)

                _vf_name = "".join(vf_name.split("_"))
                stream_set.lines.set_label(_vf_name)

        legend = plt.legend(facecolor="white")
        legend.set_zorder(ZORDER_LEGEND)

    # ==========================================================================
    # Save plotter scene as an image
    # ==========================================================================

    if save_img:

        exp_name = filename.split('/')[-1]

        cf = "".join(color_faces.split("_")) if color_faces else 0
        data_shown = f"clr{cf}_vf{int(draw_vector_fields)}_strm{int(draw_streamlines)}"
        img_name = f"{exp_name}_{data_shown}.{IMG_EXTENSION}"
        clusterer_name = mesh.attributes["clusterer_name"]
        img_path = os.path.abspath(os.path.join(DATA, "images", clusterer_name, img_name))
        plt.tight_layout()
        plotter.save(img_path, bbox_inches='tight', pad_inches=pad_inches)
        print(f"Saved image to : {img_path}")

    # ==========================================================================
    # Plot image and show to screen
    # ==========================================================================

    if show_img:
        plotter.show()

# ==============================================================================
# Main
# ==============================================================================


if __name__ == '__main__':
    fire.Fire(plot_2d)
