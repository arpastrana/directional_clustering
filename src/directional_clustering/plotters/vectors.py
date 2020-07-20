from numpy import sort
from numpy import arange
from numpy import power
from numpy import array

import matplotlib.pyplot as plt

from math import degrees

from compas.utilities import i_to_rgb

from directional_clustering.geometry import vector_from_angle
from directional_clustering.geometry import vectors_from_angles


__all__ = [
    "plot_colored_vectors"
]


def plot_colored_vectors(centers, cluster_labels, angles, name, base_vector=[1, 0, 0]):

    max_angle = max(list(angles.values()))
    centers = sort(centers.flatten())
    scales = arange(1, centers.size + 1) * 3
    scales = power(scales, 2)

    for idx, center in enumerate(centers):

        cangles = {fkey: a for fkey, a in angles.items() if cluster_labels[fkey] == center}
        vectors = vectors_from_angles(cangles, base_vector)

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


def plot_vector_field():
    return


if __name__ == "__main__":
    pass
