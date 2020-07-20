import matplotlib.pyplot as plt

from numpy import asarray
from numpy import linspace
from numpy import meshgrid
from numpy import amin
from numpy import amax
from numpy import sort

from scipy.interpolate import griddata

from directional_clustering.geometry import polygon_list_to_dict


__all__ = [
    "scalarfield_contours_numpy",
    "contour_polygons",
    "extract_polygons_from_contours"
]


def scalarfield_contours_numpy(xy, s, levels=50, density=100, method='cubic'):

    xy = asarray(xy)
    s = asarray(s)
    x = xy[:, 0]
    y = xy[:, 1]

    X, Y = meshgrid(linspace(amin(x), amax(x), 2 * density),
                    linspace(amin(y), amax(y), 2 * density))

    S = griddata((x, y), s, (X, Y), method=method)

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    c = ax.contour(X, Y, S, levels)

    contours = [0] * len(c.collections)
    levels = c.levels

    for i, coll in enumerate(iter(c.collections)):
        paths = coll.get_paths()
        contours[i] = [0] * len(paths)
        for j, path in enumerate(iter(paths)):
            polygons = path.to_polygons()
            contours[i][j] = [0] * len(polygons)
            for k, polygon in enumerate(iter(polygons)):
                contours[i][j][k] = polygon

    plt.close(fig)

    return levels, contours


def extract_polygons_from_contours(contours, levels):

    polygons = []
    for i in range(len(contours)):
        level = levels[i]
        contour = contours[i]

        for path in contour:
            for polygon in path:
                polygons.append(polygon[:-1])
    return polygons


def contour_polygons(mesh, centers, face_labels, density=100, method='nearest'):

    xy, s = [], []

    for fkey in mesh.faces():
        point = mesh.face_centroid(fkey)[:2]
        xy.append(point)
        a = face_labels[fkey]
        s.append(a)

    b = sort(centers, axis=0).flatten()

    levels, contours = scalarfield_contours_numpy(xy, s, levels=b, density=density, method=method)
    polygons = extract_polygons_from_contours(contours, levels)

    polygons = [p for p in map(polygon_list_to_dict, polygons)]
    for p in polygons:
        p['edgewidth'] = 1.0

    return polygons


if __name__ == "__main__":
    pass
