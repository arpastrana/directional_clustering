from abc import ABC
from abc import abstractmethod


__all__ = ["AbstractField"]


class AbstractField(ABC):
    """
    An abstract class for all fields.
    """
    @abstractmethod
    def dimensionality(self):
        """
        The fixed dimensionality of a field.
        """
        pass

    @abstractmethod
    def size(self):
        """
        The number of entries in the field.
        """
        pass


if __name__ == "__main__":
    pass
