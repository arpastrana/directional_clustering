import numpy as np

from compas.geometry import length_vector

from compas_plotters import MeshPlotter

from directional_clustering.plotters import line_sdl


__all__ = ["MeshPlusPlotter"]


class MeshPlusPlotter(MeshPlotter):
    """
    A 2D plotter for meshes and vector fields.
    """
    def __init__(self, *args, **kwargs):
        super(MeshPlusPlotter, self).__init__(*args, **kwargs)
        self.name = "MeshPlus Plotter"

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

            self.draw_points(points)
            # draw arrow body and retturn this as the collection
            return self.draw_lines(lines)

        return self.draw_lines(lines)
