from numpy import sort
from numpy import arange
from numpy import power
from numpy import array
from numpy import nonzero
from numpy import amax

from numpy.linalg import norm

import matplotlib.pyplot as plt

from math import degrees

from compas.utilities import i_to_rgb

from directional_clustering.geometry import vector_from_angle
from directional_clustering.geometry import vectors_from_angles


__all__ = [
    "plot_colored_vectors",
    "plot_kmeans_vectors"
]


def plot_colored_vectors(centers, cluster_labels, angles, name, base_vector=[1, 0, 0]):

    max_angle = max(list(angles.values()))  # dict fkey: angle
    centers = sort(centers.flatten())  # k-centroids
    scales = arange(1, centers.size + 1) * 3  # 3 is a factor
    scales = power(scales, 2)  # squared factors

    for idx, center in enumerate(centers):

        cangles = {fkey: a for fkey, a in angles.items() if cluster_labels[fkey] == center}
        vectors = vectors_from_angles(cangles, base_vector)  # create vectors

        X = array(list(vectors.values()))
        x = X[:, 0]
        y = X[:, 1]

        kcolor = array([[i/255.0 for i in i_to_rgb(center / max_angle)]])

        rcenter = round(center, 2)
        rdegcenter = round(degrees(center), 2)
        msg = "{} rad / {} deg".format(rcenter, rdegcenter)

        plt.scatter(x, y, c=kcolor, alpha=0.3, label=msg, s=scales[idx])

        kvector = vector_from_angle(center, base_vector)
        plt.scatter(kvector[0], kvector[1], marker="x", color="black", s=200)

    plt.axes()
    plt.grid(b=None, which='major', axis='both', linestyle='--')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("{}".format(name))
    

    plt.legend()
    plt.show()


def plot_kmeans_vectors(data, labels, centroids, normalize=False, scale_to_max=False, draw_centroids=True):
    """
    """
    if scale_to_max or normalize:
        norm_data = norm(data, axis=1, keepdims=True)
        norm_centroids = norm(centroids, axis=1, keepdims=True)

    if scale_to_max:
        max_cap = max(amax(norm_data), amax(norm_centroids))
        data = data / max_cap
        centroids = centroids / max_cap
    
    if normalize:
        data = data / norm(data, axis=1, keepdims=True)
        centroids = centroids / norm(centroids, axis=1, keepdims=True)

    for i in range(amax(labels) + 1):

        X = data[nonzero(labels==i)]
        x = X[:, 0]
        y = X[:, 1]

        # kcolor = array([[i/255.0 for i in i_to_rgb(center / max_angle)]])

        # rcenter = round(center, 2)
        # rdegcenter = round(degrees(center), 2)
        # msg = "{} rad / {} deg".format(rcenter, rdegcenter)

        # plt.scatter(x, y, c=kcolor, alpha=0.3, label=msg, s=scales[idx])
        plt.scatter(x, y, alpha=0.3)

        if draw_centroids:
            centroid = centroids[i]

            plt.scatter(centroid[0], centroid[1], marker="x", color="black", s=200)
            
            a = array([0.0, centroid[0]])
            b = array([0.0, centroid[1]])

            plt.plot(a, b, color="black", linewidth=0.75)

    # plt.axes()
    plt.grid(b=None, which='major', axis='both', linestyle='--', linewidth=0.5)

    plt.xlabel("x")
    plt.ylabel("y")
    # plt.title("{}".format(name))

    # plt.legend()
    plt.show()


if __name__ == "__main__":
    pass