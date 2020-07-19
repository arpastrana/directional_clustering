from numpy import array
from numpy import amax
from numpy import reshape
from numpy import float64

from sklearn.cluster import KMeans
from time import time


__all__ = [
    "kmeans_clustering"
    ]


def kmeans_clustering(data, n_clusters, shape=None, normalize=False, random_state=0, n_jobs=-1):

    assert isinstance(data, dict)

    np_data = [x[1] for x in data.items()]
    np_data = array(np_data, dtype=float64)

    if normalize:
        np_data /= amax(np_data)

    if reshape:
        np_data = reshape(np_data, shape)

    print()
    print("fitting {} clusters".format(n_clusters))
    t0 = time()

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_jobs=n_jobs)
    km.fit(np_data)

    print("done in {} s, number of iterations: {}".format((time() - t0), km.n_iter_))
    return km.labels_, km.cluster_centers_


if __name__ == "__main__":
    pass
