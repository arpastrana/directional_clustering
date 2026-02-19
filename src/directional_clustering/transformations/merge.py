from compas.topology import connected_components

from directional_clustering.fields import VectorField


__all__ = [
    "merge_regions",
    "merge_clusters",
    "compute_face_adjacency_clusters",
    "generate_connected_regions",
    "generate_connected_regions_adjacency",
    ]


def compute_face_adjacency_clusters(mesh, labels):
    """
    Computes the adjacency of the faces of a mesh excluding faces that belong to a different cluster.

    Parameters
    ----------
    mesh : `directional_clustering.mesh.MeshPlus`
        The mesh to compute the face adjacency for.
    labels : `dict` of `int` to `int`
        A map from a face key to the cluster label it belongs to.

    Returns
    -------
    face_adjacency : `dict` of `int` to `list` of `int`
        A map from a face key to the face keys of its neighbors within the same cluster.

    Notes
    -----
    The mesh faces must have a cluster label stored as an attribute.
    """
    assert len(labels) == mesh.number_of_faces(), "Number of labels does not equal number of faces"

    face_adjacency = {}
    for fkey in mesh.faces():
        cluster_label = labels[fkey]
        inside = []
        for okey in mesh.face_neighbors(fkey):
            if labels[okey] == cluster_label:
                inside.append(okey)
        face_adjacency[fkey] = inside

    return face_adjacency


def generate_connected_regions(adjacency):
    """
    Computes the connected regions of a mesh from an adjacency map of faces.

    Parameters
    ----------
    adjacency : `dict` of `int` to `list` of `int`
        A map from a face key to the face keys of its neighbors within the same cluster.

    Returns
    -------
    regions : `dict` of `int` to `set` of `int`
        A map from a region key to the face keys of the region.
        Each part is a connected component of the adjacency.
    """
    regions = {rkey: set(region) for rkey, region in enumerate(connected_components(adjacency))}

    return regions


def generate_connected_regions_adjacency(mesh, regions):
    """
    Generates the adjacency of the connected components of a mesh.

    Parameters
    ----------
    adjacency : `dict` of `int` to `list` of `int`
        A map from a face key to the face keys of its neighbors within the same cluster.

    Returns
    -------
    connected_components_adjacency : `dict` of `int` to `set` of `int`
        A map from a region key to the region keys of its neighbors.
    """
    # Map faces to regions
    fkeys_to_rkeys = {}
    for fkey in mesh.faces():
        for rkey, region in regions.items():
            if fkey in region:
                fkeys_to_rkeys[fkey] = rkey
                break

    # Compute part adjacencies
    region_adjacency = {rkey: set() for rkey in regions.keys()}
    for fkey, rkey in fkeys_to_rkeys.items():
        for okey in mesh.face_neighbors(fkey):
            qkey = fkeys_to_rkeys[okey]
            if rkey != qkey:
                region_adjacency[rkey].add(qkey)

    # Check that each part has at least one neighbor. No islands allowed.
    for rkey, nbrs in region_adjacency.items():
        assert len(nbrs) > 0, "Region has no neighbors"

    return region_adjacency


def compute_cluster_labels_from_regions(labels, regions):
    """
    Computes the cluster labels from the regions.

    Parameters
    ----------
    labels : `dict` of `int` to `int`
        A mapping from face key to cluster label.
    regions : `dict` of `int` to `set` of `int`
        A map from a region key to the face keys of the region.

    Returns
    -------
    pkeys_to_labels : `dict` of `int` to `int`
        A map from a region key to the cluster label.
    """
    rkeys_to_labels = {}
    for rkey, region in regions.items():
        region_labels = set(labels[fkey] for fkey in region)
        assert len(region_labels) == 1, "Region contains faces from different clusters"
        rkeys_to_labels[rkey] = region_labels.pop()

    return rkeys_to_labels


def _compute_region_area(mesh, region):
    """
    Computes the area of a region.

    Parameters
    ----------
    mesh : `directional_clustering.mesh.MeshPlus`
        The mesh to compute the area of.
    region : `set` of `int`
        A set of face keys of the region.

    Returns
    -------
    area : `float`
        The area of the region.
    """
    return sum([mesh.face_area(fkey) for fkey in region])


def merge_regions(mesh, regions, region_adjacency, min_area_ratio, max_iters=100):
    """
    Merges adjacent regions of a mesh based on their area.

    Parameters
    ----------
    mesh : `directional_clustering.mesh.MeshPlus`
        The mesh to merge the regions of.
    parts : `dict` of `int` to `set` of `int`
        A map from a region key to the face keys of the region.
    part_adjacency : `dict` of `int` to `set` of `int`
        A map from a part key to the part keys of its neighbors.
    min_area_ratio : `float`
        The minimum area ratio of a part to merge it with its neighbor.
    max_iters : `int`, optional
        The maximum number of iterations to perform. Defaults to 100.

    Returns
    -------
    new_parts : `dict` of `int` to `set` of `int`
        A map from a region key to the face keys of the region.
    """
    total_area = mesh.area()
    assert min_area_ratio < 1.0, "Minimum area ratio must be less than 1.0"
    min_area = min_area_ratio * total_area

    regions = {k: v for k, v in regions.items()}
    region_adjacency = {k: v for k, v in region_adjacency.items()}

    for _ in range(max_iters):
        rkeys_areas = {rkey: _compute_region_area(mesh, regions[rkey]) for rkey in regions.keys()}
        rkeys = [rkey for rkey in regions.keys() if rkeys_areas[rkey] <= min_area]

        if len(rkeys) == 0:
            return regions

        sorted_rkeys = sorted(rkeys, key=lambda x: rkeys_areas[x])

        for rkey in sorted_rkeys:
            nbrs = region_adjacency[rkey]
            # TODO: Use smallest or largest neighbor?
            # smallest_nbr = min(nbrs, key=lambda x: rkeys_areas[x])
            nbr_to_join = max(nbrs, key=lambda x: rkeys_areas[x])

            # Merge region faces into neighbor's faces
            region = regions[rkey]
            regions[nbr_to_join].update(region)
            for _rkey in region_adjacency[rkey]:
                if _rkey != nbr_to_join:
                    region_adjacency[nbr_to_join].add(_rkey)

            # Recompute part adjacency
            for nbr in nbrs:
                region_adjacency[nbr].remove(rkey)
                if nbr != nbr_to_join:
                    region_adjacency[nbr].add(nbr_to_join)

            # Delete region
            del regions[rkey]
            del region_adjacency[rkey]
            break

    print("Warning: Exceeded max number of iterations. Returning current regions.")
    return regions


def compute_face_clusters_from_regions(regions, rkeys_to_labels):
    """
    Computes the cluster labels of the faces of a mesh from the regions.

    Parameters
    ----------
    parts : `dict` of `int` to `set` of `int`
        A map from a region key to the face keys of the region.
    rkeys_to_labels : `dict` of `int` to `int`
        A map from a region key to the cluster label.

    Returns
    -------
    labels : `dict` of `int` to `int`
        A map from a face key to the cluster label.
    """
    labels = {}
    for rkey, region in regions.items():
        label = rkeys_to_labels[rkey]
        for fkey in region:
            labels[fkey] = label

    return labels


def compute_vector_field_from_clusters(labels, cluster_centroids):
    """
    Creates a face-based vector field from cluster assignments.

    Parameters
    ----------
    labels : `dict` of `int` to `int`
        A mapping from face key to cluster label.
    cluster_centroids : `dict` of `int` to sequence of `float`
        A mapping from cluster label to centroid vector.

    Returns
    -------
    vector_field : `directional_clustering.fields.VectorField`
        A vector field where each face vector is the centroid of its assigned cluster.
    """
    vector_field = VectorField()
    for fkey, label in labels.items():
        vector = cluster_centroids[label]
        vector_field.add_vector(fkey, vector)

    return vector_field


def merge_clusters(mesh, labels, centroids=None, min_area_ratio=0.01, max_iters=100):
    """
    Merges small connected cluster regions on a mesh.

    Parameters
    ----------
    mesh : `directional_clustering.mesh.MeshPlus`
        The mesh whose face clusters are merged.
    labels : `dict` of `int` to `int`
        A mapping from face key to cluster label.
    centroids : `dict` of `int` to sequence of `float`, optional
        Cluster centroid vectors keyed by cluster label.
        If provided, a vector field consistent with the merged labels is also returned.
    min_area_ratio : `float`, optional
        Minimum region area threshold as a fraction of the total mesh area.
        Regions below this threshold are iteratively merged into adjacent regions.
        Defaults to ``0.01``.
    max_iters : `int`, optional
        Maximum number of merge iterations. Defaults to ``100``.

    Returns
    -------
    labels : `dict` of `int` to `int`
        Updated mapping from face key to merged cluster label.
    tuple : (`dict`, `directional_clustering.fields.VectorField`)
        Returned when ``centroids`` is provided; contains merged labels and the
        corresponding face-based vector field.
    """
    assert len(labels) == mesh.number_of_faces(), "Number of labels does not equal number of faces"

    face_adjacency = compute_face_adjacency_clusters(mesh, labels)
    regions = generate_connected_regions(face_adjacency)
    for rkey, region in regions.items():
        print(f"Region {rkey} has area ratio {_compute_region_area(mesh, region) / mesh.area():.3f}")

    region_adjacency = generate_connected_regions_adjacency(mesh, regions)

    # Compute cluster labels from regions
    rkeys_to_labels = compute_cluster_labels_from_regions(labels, regions)
    new_regions = merge_regions(mesh, regions, region_adjacency, min_area_ratio, max_iters)

    if len(new_regions) < len(regions):
        labels = compute_face_clusters_from_regions(new_regions, rkeys_to_labels)
        print(f"Merged {len(regions) - len(new_regions)} regions!")
    else:
        print("No regions merged...")

    # Merged data
    if centroids is not None:
        vector_field = compute_vector_field_from_clusters(labels, centroids)
        return labels, vector_field

    return labels


if __name__ == "__main__":
    from directional_clustering.mesh import MeshPlus
    import matplotlib.pyplot as plt
    from compas_plotters import Plotter
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    mesh = MeshPlus.from_meshgrid(dx=5, nx=5)


    clusters = {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        1: [14, 19],
        2: [20, 21, 22, 23],
        3: [10, 11, 12, 13, 15, 16, 17, 18, 24]
    }

    # Set cluster labels on mesh
    for ckey, fkeys in clusters.items():
        for fkey in fkeys:
            mesh.face_attribute(fkey, "cluster", ckey)

    # Data extraction
    labels = mesh.cluster_labels("cluster")

    # Merging?
    min_area_ratio = 9 / 25.0
    labels = merge_clusters(mesh, labels, min_area_ratio=min_area_ratio)

    # Visualization
    n_clusters = len(clusters)
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
