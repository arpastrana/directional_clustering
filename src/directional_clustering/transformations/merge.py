from compas.datastructures import Mesh
from compas.topology import connected_components


# __all__ = [
#     "merge_clusters",
#     "merge_clusters_labels",
#     "merge_clustered_field",
#     ]


# def explode_mesh(mesh, min_num_faces=1):
#     """
#     Explode a mesh into all its disjoint meshes using the connected components of the face adjacency.

#     Parameters
#     ----------
#     mesh : `compas.datastructures.Mesh`
#         The mesh to explode.
#     min_num_faces : `int`, optional
#         The minimum number of faces to consider a connected component.
#         Defaults to `1`.

#     Returns
#     -------
#     exploded_meshes : `list` of `compas.datastructures.Mesh`
#         The exploded meshes.
#     """
#     face_adjacency = {fkey: mesh.face_neighbors(fkey) for fkey in mesh.faces()}
#     parts = [set(part) for part in connected_components(face_adjacency)]

#     cls = type(mesh)
#     exploded_meshes = []
#     for part in parts:

#         part = set(part)
#         if len(part) < min_num_faces:
#             continue

#         vertex_keys = list(set([vkey for fkey in part for vkey in mesh.face_vertices(fkey)]))
#         vertices = [mesh.vertex_coordinates(vkey) for vkey in vertex_keys]
#         key_to_index = {vkey: i for i, vkey in enumerate(vertex_keys)}
#         faces = [[key_to_index[vkey] for vkey in mesh.face_vertices(fkey)] for fkey in part]
#         exploded_meshes.append(cls.from_vertices_and_faces(vertices, faces))

#     return exploded_meshes


# def compute_k_meshes(mesh, cluster_keys, face_labels):
#     """
#     Generates a mesh per directional cluster by grouping faces based on their cluster labels.

#     Parameters
#     ----------
#     mesh : `directional_clustering.mesh.MeshPlus`
#         The mesh to compute the k-meshes from.
#     cluster_keys : `list` of `int`
#         The keys of the clusters to compute the meshes for.
#     face_labels : `dict` of `int` to `int`
#         A map from a face key to its cluster label.

#     Returns
#     -------
#     k_meshes : `dict` of `int` to `directional_clustering.mesh.MeshPlus`
#         A map from a cluster key to its mesh.
#     """
#     k_meshes = {}

#     for ckey in cluster_keys:
#         k_fkeys = [fkey for fkey, c in face_labels.items() if c == ckey]
#         k_polygons = [mesh.face_coordinates(fkey) for fkey in k_fkeys]
#         k_mesh = Mesh.from_polygons(k_polygons)
#         k_meshes[ckey] = k_mesh

#     return k_meshes


# def compute_k_meshes_disjoint(mesh, cluster_keys, face_labels):
#     """
#     Generates a list of disjoint meshes per directional cluster.
#     The meshes in a cluster are obtained by based on the connected components of the face adjacency.

#     Parameters
#     ----------
#     mesh : `directional_clustering.mesh.MeshPlus`
#         The mesh to compute the k-meshes from.
#     cluster_keys : `list` of `int`
#         The keys of the clusters to compute the meshes for.
#     face_labels : `dict` of `int` to `int`
#         A map from a face key to its cluster label.

#     Returns
#     -------
#     k_meshes_disjoint : `dict` of `int` to `list` of `compas.datastructures.Mesh`
#         A map from a cluster key to its list of disjoint meshes.
#     """
#     k_meshes_disjoint = {}
#     k_meshes = compute_k_meshes(mesh, cluster_keys, face_labels)
#     for ckey, k_mesh in k_meshes.items():
#         k_meshes_disjoint[ckey] = explode_mesh(k_mesh)

#     return k_meshes_disjoint


# def merge_clusters(mesh, cluster_keys, face_labels, min_area_ratio=0.05):
#     """
#     Merges small clusters into their smallest neighboring cluster if their area is smaller than the given threshold.

#     Parameters
#     ----------
#     mesh : `directional_clustering.mesh.MeshPlus`
#         The mesh to merge the clusters of.
#     min_area_ratio : `float`
#         The minimum area ratio of a disjoint cluster mesh to be considered for merging.
#         Defaults to `0.05`.

#     Returns
#     -------
#     """
#     mesh = mesh.copy()
#     k_meshes_disjoint = compute_k_meshes_disjoint(mesh, cluster_keys, face_labels)

#     return mesh

def compute_face_adjacency_per_cluster_from_mesh(mesh):
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

    # Check that all faces have a cluster label
    for fkey in mesh.faces():
        assert mesh.face_attribute(fkey, "cluster") is not None

    # Check sum of cluster areas equals total area of mesh
    total_area = mesh.area()
    cluster_areas = {ckey: sum([mesh.face_area(fkey) for fkey in fkeys]) for ckey, fkeys in clusters.items()}
    assert sum(cluster_areas.values()) == total_area, "Sum of cluster areas does not equal total area of mesh"

    labels = mesh.cluster_labels("cluster")
    n_clusters = len(clusters)

    # Check sum of cluster meshes equals total area of mesh
    k_meshes = compute_k_meshes(mesh, clusters.keys(), labels)
    assert sum([k_mesh.area() for k_mesh in k_meshes.values()]) == total_area, "Sum of cluster meshes does not equal total area of mesh"

    # Check sum of cluster disjoint meshes equals total area of mesh
    k_meshes_disjoint = compute_k_meshes_disjoint(mesh, clusters.keys(), labels)
    areas = []
    for ckey, k_meshes in k_meshes_disjoint.items():
        for k_mesh in k_meshes:
            areas.append(k_mesh.area())
    assert sum(areas) == total_area, "Sum of cluster disjoint meshes does not equal total area of mesh"

    # Merge clusters

    # Compute face adjacencies
    face_adjacency_inside = {}
    for fkey in mesh.faces():
        cluster_label = mesh.face_attribute(fkey, "cluster")
        inside = []
        for okey in mesh.face_neighbors(fkey):
            if mesh.face_attribute(okey, "cluster") == cluster_label:
                inside.append(okey)
        face_adjacency_inside[fkey] = inside

    # Find connected components per cluster
    parts = {pkey: set(part) for pkey, part in enumerate(connected_components(face_adjacency_inside))}
    assert len(parts) == len(parts_truth), "Number of parts does not match number of parts truth"
    assert all(part in parts_truth for part in parts.values()), "Parts do not match parts truth"

    # Map parts to cluster labels
    pkeys_to_labels = {}
    for pkey, part in parts.items():
        part_labels = set(mesh.face_attribute(fkey, "cluster") for fkey in part)
        assert len(part_labels) == 1, "Part contains faces from different clusters"
        pkeys_to_labels[pkey] = part_labels.pop()

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

    for pkey, nbrs in part_adjacency.items():
        assert len(nbrs) > 0, "Part has no neighbors"

    # Updated parts
    print("Initial parts:")
    for pkey, part in parts.items():
        print(pkey, part)

    print("Initial part adjacencies:")
    for pkey, nbrs in part_adjacency.items():
        print(pkey, nbrs)

    # Merging?
    def polygons_area(mesh, part):
        return sum([mesh.face_area(fkey) for fkey in part])

    min_area_ratio = 3 / 25.0
    total_area = mesh.area()
    assert min_area_ratio < 1.0, "Minimum area ratio must be less than 1.0"
    min_area = min_area_ratio * total_area

    # Probably loop starts here
    # small_parts_exist = True
    num_iters = 100
    for i in range(num_iters):
        print(f"***Looping starts with {len(parts)} parts***")

        pkeys_areas = {pkey: polygons_area(mesh, parts[pkey]) for pkey in parts.keys()}
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
            # breakpoint()
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

    # Updated parts
    print(f"\nNew number of parts: {len(parts)}")
    print("Updated parts:")
    for pkey, part in parts.items():
        print(pkey, part)

    print("Updated part adjacencies:")
    for pkey, nbrs in part_adjacency.items():
        print(pkey, nbrs)

    # Update face labels
    labels_ = {}
    for pkey, part in parts.items():
        label = pkeys_to_labels[pkey]
        for fkey in part:
            labels_[fkey] = label
    labels = labels_
    # mesh.cluster_labels("cluster", labels)

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
