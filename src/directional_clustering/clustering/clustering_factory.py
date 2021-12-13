from directional_clustering.clustering import ClusteringAlgorithm
from directional_clustering.clustering import CosineKMeans
from directional_clustering.clustering import CosineAbsoluteKMeans
from directional_clustering.clustering import VariationalKMeans
from directional_clustering.clustering import EuclideanKMeans
from directional_clustering.clustering import DifferentiableCosineKMeans
from directional_clustering.clustering import DifferentiableCosineAbsoluteKMeans


__all__ = ["ClusteringFactory"]


class ClusteringFactory(object):
    """
    A factory to unify the creation of clustering algorithms.
    """
    supported_algorithms = {}

    @classmethod
    def create(cls, name):
        """
        Creates an unitialized clustering algorithm.

        Parameters
        ----------
        name : `str`
            The name of the clustering algorithm to generate.

        Returns
        -------
        algorithm : `directional_clustering.clustering.ClusteringAlgorithm`
            A clustering algorithm to instantiate.
        """
        algorithm = cls.supported_algorithms.get(name)

        if algorithm is None:
            raise KeyError(f"Algorithm {name} is not supported!")

        return algorithm

    @classmethod
    def register(cls, name, algorithm):
        """
        Registers a clustering algorithm to the factory's database.

        Parameters
        ----------
        name : `str`
            The name key by which a clustering will be stored.
        algorithm : `directional_clustering.clustering.ClusteringAlgorithm`
            A clustering algorithm.
        """
        assert isinstance(algorithm, type(ClusteringAlgorithm))
        cls.supported_algorithms[name] = algorithm


# Register supported algorithms
ClusteringFactory.register("cosine_kmeans", CosineKMeans)
ClusteringFactory.register("cosine_kmeans_abs", CosineAbsoluteKMeans)
ClusteringFactory.register("cosine_kmeans_diff", DifferentiableCosineKMeans)
ClusteringFactory.register("cosine_kmeans_abs_diff", DifferentiableCosineAbsoluteKMeans)
ClusteringFactory.register("variational_kmeans", VariationalKMeans)
ClusteringFactory.register("euclidean_kmeans", EuclideanKMeans)

if __name__ == "__main__":
    pass
