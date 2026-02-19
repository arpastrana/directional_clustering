import pytest

from directional_clustering.mesh import MeshPlus

from directional_clustering.transformations import compute_face_adjacency_clusters
from directional_clustering.transformations import generate_connected_regions_adjacency
from directional_clustering.transformations import generate_connected_regions
from directional_clustering.transformations import merge_regions


@pytest.fixture
def mesh():
    """
    """
    return MeshPlus.from_meshgrid(dx=5, nx=5)


@pytest.fixture
def regions_truth():
    """
    """
    return [
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {14, 19},
        {20, 21, 22, 23},
        {10, 11, 12, 13, 15, 16, 17, 18},
        {24}
    ]


@pytest.fixture
def regions_adjacency_truth():
    """
    """
    return {0: {1, 2}, 1: {0, 2, 3}, 2: {0, 1, 4}, 3: {1, 4}, 4: {2, 3}}


@pytest.fixture
def cluster_labels():
    """
    """
    return {0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            1: [14, 19],
            2: [20, 21, 22, 23],
            3: [10, 11, 12, 13, 15, 16, 17, 18, 24]}


@pytest.fixture
def labels(cluster_labels):
    """
    """
    labels = {}
    for ckey, fkeys in cluster_labels.items():
        for fkey in fkeys:
            labels[fkey] = ckey
    return labels


def generate_regions(mesh, labels):
    adjacency = compute_face_adjacency_clusters(mesh, labels)
    return generate_connected_regions(adjacency)


def generate_regions_adjacency(mesh, labels):
    regions = generate_regions(mesh, labels)
    return generate_connected_regions_adjacency(mesh, regions)


def test_regions_generation(mesh, labels, regions_truth):
    regions = generate_regions(mesh, labels)
    assert len(regions) == len(regions_truth)
    assert all(region in regions_truth for region in regions.values())


def test_regions_adjacency(mesh, labels, regions_adjacency_truth):
    regions_adjacency = generate_regions_adjacency(mesh, labels)
    assert regions_adjacency == regions_adjacency_truth


@pytest.mark.parametrize("min_area_ratio, result", ([0.0, 5], [1.0, 4], [3.0, 3], [7.0, 2]))
def test_merge_regions_area(mesh, labels, min_area_ratio, result):
    regions = generate_regions(mesh, labels)
    regions_adjacency = generate_regions_adjacency(mesh, labels)

    min_area_ratio = min_area_ratio / mesh.area()
    regions = merge_regions(mesh, regions, regions_adjacency, min_area_ratio=min_area_ratio)
    assert len(regions) == result

    assert sum([len(region) for region in regions.values()]) == mesh.number_of_faces()

    total_area = 0.0
    for region in regions.values():
        region_area = sum([mesh.face_area(fkey) for fkey in region])
        total_area += region_area
    assert total_area == mesh.area()


def test_all_faces_included(mesh, labels):
    regions = generate_regions(mesh, labels)
    regions_adjacency = generate_regions_adjacency(mesh, labels)
    for i in range(10):
        min_area_ratio = i / 100.0
        regions = merge_regions(mesh, regions, regions_adjacency, min_area_ratio=min_area_ratio)
        assert set(mesh.faces()) == set().union(*regions.values())
