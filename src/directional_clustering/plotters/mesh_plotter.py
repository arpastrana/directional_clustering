from compas.geometry import length_vector

from compas_plotters import MeshPlotter

from directional_clustering.plotters import line_sdl
from directional_clustering.plotters import line_tuple_to_dict


__all__ = ["MeshPlusPlotter"]


class MeshPlusPlotter(MeshPlotter):
    """
    A 2D plotter for meshes and vector fields.
    """
    def __init__(self, *args, **kwargs):
        super(MeshPlusPlotter, self).__init__(*args, **kwargs)
        self.name = "MeshPlus Plotter"

    def draw_vector_field(self, field, color, uniform, scale, width=0.5, as_arrows=False):
        """
        Draws a vector field on a mesh.

        It assumes the field and the mesh faces have the same keys.
        """
        lines = []
        mesh = self.mesh

        both_sides = True
        if as_arrows:
            both_sides = False

        for fkey in range(field.size()):
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
            line["color"] = color

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
