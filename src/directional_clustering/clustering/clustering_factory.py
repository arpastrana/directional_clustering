from directional_clustering.clustering import CosineKMeans
from directional_clustering.clustering import VariationalKMeans


__all__ = ["ClusteringFactory"]


class ClusteringFactory(object):
    """
    A factory to unify the creation of supporting clustering algorithms.
    """
    supported_algorithms = {}

    @classmethod
    def create(cls, clustering_name):
        """
        Create a clustering algorithm.
        """
        algorithm = cls.supported_algorithms.get(clustering_name)

        if algorithm is None:
            raise KeyError(f"Algorithm {clustering_name} is not supported!")

        return algorithm

    @classmethod
    def register(cls, name, algorithm):
        """
        Register a clustering algorithm.
        """
        cls.supported_algorithms[name] = algorithm


# Register supported algorithms
ClusteringFactory.register("cosine kmeans", CosineKMeans)
ClusteringFactory.register("variational kmeans", VariationalKMeans)


if __name__ == "__main__":
    # Small tests
    print(ClusteringFactory.supported_algorithms)
    print(ClusteringFactory.create("cosine kmeans"))
    print(ClusteringFactory.create("variational kmeans"))
    print(ClusteringFactory.create("unsupported clustering"))
