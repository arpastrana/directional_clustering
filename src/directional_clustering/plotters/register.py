from compas.artists import Artist

from directional_clustering.mesh import MeshPlus
from directional_clustering.plotters import MeshPlusArtist


__all__ = ["register_artists"]


def register_artists():
    """
    Register objects to the artist factory.
    """
    Artist.register(MeshPlus, MeshPlusArtist, context="Plotter")
