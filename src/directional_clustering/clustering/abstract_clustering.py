from abc import ABC
from abc import abstractmethod
from abc import abstractproperty


__all__ = ["ClusteringAlgorithm"]


class ClusteringAlgorithm(ABC):
    """
    Abstract base class for all clustering algorithms.
    """
    @abstractmethod
    def cluster(self, *args, **kwargs):
        """
        Main clustering method.
        """
        pass

    @abstractproperty
    def loss(self):
        """
        The final error of the produced by the clustering method.
        """
        pass

    @abstractproperty
    def clustered_field(self):
        """
        The clustered vector field.
        """
        pass

    @abstractproperty
    def labels(self):
        """
        The labels that reference entries in the vector field to clusters.
        """
        pass
