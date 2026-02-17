import numpy as np

from compas.artists import Artist

from compas.geometry import length_vector

from compas_plotters import draw_xpoints_xy
from compas_plotters import draw_xlines_xy
from compas_plotters.artists import MeshArtist

from directional_clustering.mesh import MeshPlus
from directional_clustering.plotters import line_sdl


__all__ = ["MeshPlusArtist"]


class MeshPlusArtist(MeshArtist):
    """
    A 2D artist for meshes and vector fields.
    """
    def __init__(self, *args, **kwargs):
        super(MeshPlusArtist, self).__init__(*args, **kwargs)
        self.name = "MeshPlusPlotter"

    def draw_vector_field(self,
                          field,
                          color,
                          uniform,
                          scale,
                          width=0.5,
                          ratio=1.0,
                          seed=None,
                          as_arrows=False):
        """
        Draws a vector field on a mesh.

        It assumes the field and the mesh faces have the same keys.
        """
        assert ratio > 0.0

        np.random.seed(seed)

        lines = []
        mesh = self.mesh

        both_sides = True
        if as_arrows:
            both_sides = False

        # TODO: we assume the fkeys are ordered ints in the range [0, field.size()]
        fkeys = list(range(field.size()))

        if ratio < 1.0:
            fkeys = np.random.choice(fkeys, size=int(field.size() * ratio), replace=False)

        for fkey in fkeys:
            line = {}
            vector = field[fkey]
            length = scale

            if not uniform:
                length = length_vector(vector) * scale

            start, end = line_sdl(mesh.face_centroid(fkey), vector, length, both_sides)

            line["start"] = start
            line["end"] = end
            line["length"] = length
            line["width"] = width

            # duck typing with color, assumes dict of color, defaults to single color
            if isinstance(color, dict):
                clr = color[fkey]
            else:
                clr = color
            line["color"] = clr

            lines.append(line)

        if as_arrows:
            # draw arrow heads as points
            points = []
            for line in lines:
                point = {}
                point["pos"] = line["end"]
                point["radius"] = line["length"] * 0.25
                point["facecolor"] = color
                point["edgecolor"] = color
                point["edgewidth"] = 0.001

                points.append(point)

            draw_xpoints_xy(points, self.plotter.axes)

            # draw arrow body and return this as the collection
            return draw_xlines_xy(lines, self.plotter.axes)

        return draw_xlines_xy(lines, self.plotter.axes)


# Register artist
Artist.register(MeshPlus, MeshPlusArtist, context="Plotter")
