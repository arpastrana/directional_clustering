from compas.artists import Artist
from compas.plugins import plugin

from directional_clustering.mesh import MeshPlus
from directional_clustering.plotters import MeshPlusArtist


__all__ = ["register_artists"]


@plugin(category="factories", requires=["matplotlib"])
def register_artists():
    """
    Register objects to the artist factory.
    """
    Artist.register(MeshPlus, MeshPlusArtist, context="Plotter")
