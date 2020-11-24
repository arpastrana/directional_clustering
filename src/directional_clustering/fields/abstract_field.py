from abc import ABC
from abc import abstractmethod


__all__ = ["AbstractField"]


class AbstractField(ABC):
    """
    An abstract field.
    """
    @abstractmethod
    def dimensionality(self):
        """
        The fixed dimensionality of a field.
        """
        return

    @abstractmethod
    def size(self):
        """
        The number of entries in the field.
        """
        return


if __name__ == "__main__":
    pass
