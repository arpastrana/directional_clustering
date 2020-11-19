from abc import ABC
from abc import abstractmethod


__all__ = ["ClusteringAlgorithm"]


class ClusteringAlgorithm(ABC):
    """
    Abstract base class for all clustering algorithms.
    """
    def __init__(self):
        # these may be better off as properties
        self.loss = None
        self.centroids = None
        self.labels = None

    @abstractmethod
    def cluster(self):
        """
        """
        return
