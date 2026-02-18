from compas.topology import connected_components

from directional_clustering.fields import VectorField


# __all__ = [
#     "merge_clusters",
#     "merge_clusters_labels",
#     "merge_clustered_field",
#     ]


def compute_face_adjacency_per_cluster(mesh):
    """
    Computes the adjacency of the faces of a mesh excluding faces that belong to a different cluster.

    Parameters
    ----------
    mesh : `directional_clustering.mesh.MeshPlus`
        The mesh to compute the face adjacency for.

    Returns
    -------
    face_adjacency : `dict` of `int` to `list` of `int`
        A map from a face key to the face keys of its neighbors within the same cluster.

    Notes
    -----
    The mesh faces must have a cluster label stored as an attribute.
    """
    # Compute face adjacencies
    face_adjacency = {}
    for fkey in mesh.faces():
        cluster_label = mesh.face_attribute(fkey, "cluster")
        inside = []
        for okey in mesh.face_neighbors(fkey):
            if mesh.face_attribute(okey, "cluster") == cluster_label:
                inside.append(okey)
        face_adjacency[fkey] = inside

    return face_adjacency


def generate_connected_parts(adjacency):
    """
    Generates connected components from an adjacency.

    Parameters
    ----------
    adjacency : `dict` of `int` to `list` of `int`
        A map from a element keys to the neighbors of each element key.

    Returns
    -------
    parts : `dict` of `int` to `set` of `int`
        A map from a part key to the element keys of the part.
        Each part is a connected component of the adjacency.
    """
    parts = {pkey: set(part) for pkey, part in enumerate(connected_components(adjacency))}

    return parts


def generate_connected_parts_adjacency(mesh, parts):
    """
    Generates the adjacency of the connected components of a mesh.

    Parameters
    ----------
    adjacency : `dict` of `int` to `list` of `int`
        A map from a element keys to the neighbors of each element key.

    Returns
    -------
    connected_components_adjacency : `dict` of `int` to `set` of `int`
        A map from a part key to the part keys of its neighbors.
    """
    # Map faces to parts
    fkeys_to_pkeys = {}
    for fkey in mesh.faces():
        for pkey, part in parts.items():
            if fkey in part:
                fkeys_to_pkeys[fkey] = pkey
                break

    # Compute part adjacencies
    part_adjacency = {pkey: set() for pkey in parts.keys()}
    for fkey, pkey in fkeys_to_pkeys.items():
        for okey in mesh.face_neighbors(fkey):
            qkey = fkeys_to_pkeys[okey]
            if pkey != qkey:
                part_adjacency[pkey].add(qkey)

    # Check that each part has at least one neighbor. No islands allowed.
    for pkey, nbrs in part_adjacency.items():
        assert len(nbrs) > 0, "Part has no neighbors"

    return part_adjacency


def compute_cluster_labels_from_parts(mesh, parts):
    """
    Computes the cluster labels from the parts.

    Parameters
    ----------
    mesh : `directional_clustering.mesh.MeshPlus`
        The mesh to compute the cluster labels from.
    parts : `dict` of `int` to `set` of `int`
        A map from a part key to the face keys of the part.

    Returns
    -------
    pkeys_to_labels : `dict` of `int` to `int`
        A map from a part key to the cluster label.
    """
    pkeys_to_labels = {}
    for pkey, part in parts.items():
        part_labels = set(mesh.face_attribute(fkey, "cluster") for fkey in part)
        assert len(part_labels) == 1, "Part contains faces from different clusters"
        pkeys_to_labels[pkey] = part_labels.pop()

    return pkeys_to_labels


def _compute_faces_area(mesh, part):
    """
    """
    return sum([mesh.face_area(fkey) for fkey in part])


def merge_parts(mesh, parts, part_adjacency, min_area_ratio, max_iters=100):
    """
    Merges parts of a mesh based on their area.

    Parameters
    ----------
    mesh : `directional_clustering.mesh.MeshPlus`
        The mesh to merge the parts of.
    parts : `dict` of `int` to `set` of `int`
        A map from a part key to the face keys of the part.
    part_adjacency : `dict` of `int` to `set` of `int`
        A map from a part key to the part keys of its neighbors.
    min_area_ratio : `float`
        The minimum area ratio of a part to merge it with its smallest neighbor.
    max_iters : `int`, optional
        The maximum number of iterations to perform. Defaults to 100.

    Returns
    -------
    new_parts : `dict` of `int` to `set` of `int`
        A map from a part key to the face keys of the part.
    """
    total_area = mesh.area()
    assert min_area_ratio < 1.0, "Minimum area ratio must be less than 1.0"
    min_area = min_area_ratio * total_area

    parts = {k: v for k, v in parts.items()}
    part_adjacency = {k: v for k, v in part_adjacency.items()}

    for _ in range(max_iters):
        print(f"***Looping starts with {len(parts)} parts***")

        pkeys_areas = {pkey: _compute_faces_area(mesh, parts[pkey]) for pkey in parts.keys()}
        pkeys = [pkey for pkey in parts.keys() if pkeys_areas[pkey] <= min_area]

        if len(pkeys) == 0:
            break

        sorted_pkeys = sorted(pkeys, key=lambda x: pkeys_areas[x])

        for pkey in sorted_pkeys:
            print(f"Processing part {pkey}")

            # Find smallest neighbor
            nbrs = part_adjacency[pkey]
            smallest_nbr = min(nbrs, key=lambda x: pkeys_areas[x])
            print(f"\tMerging part {pkey} with area {pkeys_areas[pkey]} into smallest neighbor {smallest_nbr}")

            # Merge part faces into smallest neighbor's faces
            part = parts[pkey]
            parts[smallest_nbr].update(part)
            print(f"\tUpdated part {smallest_nbr} with part {part}")
            for _pkey in part_adjacency[pkey]:
                if _pkey != smallest_nbr:
                    part_adjacency[smallest_nbr].add(_pkey)
            print(f"\tUpdated part adjacency of {smallest_nbr} with adjacency {part_adjacency[pkey]}")

            # Recompute part adjacency
            for nbr in nbrs:
                part_adjacency[nbr].remove(pkey)
                if nbr != smallest_nbr:
                    part_adjacency[nbr].add(smallest_nbr)

            # Delete part
            del parts[pkey]
            del part_adjacency[pkey]
            break

        # print(f"Any changes: {any_changes}")
        print("\tIntermediate part adjacencies:")
        for pkey, nbrs in part_adjacency.items():
            print(f"\t{pkey}: {nbrs}")
        print("***Looping ends***")

    return parts


def compute_face_clusters_from_parts(parts, pkeys_to_labels):
    """
    Computes the cluster labels of the faces of a mesh from the parts.

    Parameters
    ----------
    parts : `dict` of `int` to `set` of `int`
        A map from a part key to the face keys of the part.
    pkeys_to_labels : `dict` of `int` to `int`
        A map from a part key to the cluster label.

    Returns
    -------
    labels : `dict` of `int` to `int`
        A map from a face key to the cluster label.
    """
    labels = {}
    for pkey, part in parts.items():
        label = pkeys_to_labels[pkey]
        for fkey in part:
            labels[fkey] = label

    return labels


def compute_vector_field_from_clusters(labels, cluster_centroids):
    """
    Computes the vector field from the clusters.
    """
    vector_field = VectorField()
    for fkey, label in labels.items():
        vector = cluster_centroids[label]
        vector_field.add_vector(fkey, vector)
    return vector_field


def merge_clusters(mesh, min_area_ratio, max_iters=100):
    """
    Merges clusters of a mesh in geometry space based on the cluster'sarea.
    """
    face_adjacency = compute_face_adjacency_per_cluster(mesh)
    parts = generate_connected_parts(face_adjacency)
    part_adjacency = generate_connected_parts_adjacency(mesh, parts)
    pkeys_to_labels = compute_cluster_labels_from_parts(mesh, parts)

    new_parts = merge_parts(mesh, parts, part_adjacency, min_area_ratio, max_iters)

    # Merged data
    labels = compute_face_clusters_from_parts(new_parts, pkeys_to_labels)
    # vector_field = compute_vector_field_from_clusters(labels, clustered_field)

    # TODO: Check all faces are included in the new parts
    # TODO: Check sum of cluster meshes equals total area of mesh
    # TODO: Check sum of cluster disjoint meshes equals total area of mesh

    return labels  # , vector_field


if __name__ == "__main__":
    from directional_clustering.mesh import MeshPlus
    import matplotlib.pyplot as plt
    from compas_plotters import Plotter
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    mesh = MeshPlus.from_meshgrid(dx=5, nx=5)

    parts_truth = [
        {0, 1, 2, 3, 4, 6, 7, 8, 9},
        {14, 19},
        {20, 21, 22, 23},
        {5, 10, 11, 12, 13, 15, 16, 17, 18},
        {24}
    ]

    clusters = {
        0: [0, 1, 2, 3, 4, 6, 7, 8, 9],
        1: [14, 19],
        2: [20, 21, 22, 23],
        3: [5, 10, 11, 12, 13, 15, 16, 17, 18, 24]
    }

    # Set cluster labels on mesh
    for ckey, fkeys in clusters.items():
        for fkey in fkeys:
            mesh.face_attribute(fkey, "cluster", ckey)

    # Data extraction
    labels = mesh.cluster_labels("cluster")
    n_clusters = len(clusters)

    # Check sum of cluster areas equals total area of mesh
    total_area = mesh.area()
    cluster_areas = {ckey: sum([mesh.face_area(fkey) for fkey in fkeys]) for ckey, fkeys in clusters.items()}
    assert sum(cluster_areas.values()) == total_area, "Sum of cluster areas does not equal total area of mesh"

    # Merging?
    # TODO: Reduce dependency on mesh, input labels and vector field instead
    # Do not assume the mesh has a cluster attribute and a vector field attribute
    min_area_ratio = 2 / 25.0
    labels = merge_clusters(mesh, min_area_ratio)

    # Visualization
    cmap = plt.cm.get_cmap("rainbow", n_clusters)
    normalize = Normalize(0, n_clusters - 1)
    sm = ScalarMappable(normalize, cmap)
    face_colors = {key: sm.to_rgba(label)[:-1] for key, label in labels.items()}
    face_text = {fkey: str(fkey) for fkey in mesh.faces()}

    plotter = Plotter(figsize=(8, 8))
    artist = plotter.add(mesh, facecolor=face_colors, show_vertices=False)
    artist.draw_facelabels(face_text)
    plotter.zoom_extents()
    plotter.show()
