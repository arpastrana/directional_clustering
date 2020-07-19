from math import fabs
from time import time

from directional_clustering.clusters import faces_labels
from directional_clustering.clusters import kmeans_clustering


__all__ = [
    "kmeans_errors"
    ]


def kmeans_errors(mesh, data, max_clusters, early_stopping, shape, eps):

    errors = []
    
    for n_clusters in range(1, max_clusters + 1):
        k_error = 0.0

        labels, centers = kmeans_clustering(data, n_clusters, shape=shape)
        cluster_labels = faces_labels(mesh, labels, centers)

        for fkey in mesh.faces():
            c_angle = cluster_labels[fkey]
            angle = data[fkey]

            error = (angle - c_angle) ** 2
            k_error += error

        k_error /= len(cluster_labels.values())
        errors.append(k_error)

        if n_clusters < 2 or not early_stopping:
            continue

        delta_error = (errors[-2] - errors[-1]) / errors[-1]
        if fabs(delta_error) < eps:
            break
    
    return errors


if __name__ == "__main__":
    pass
