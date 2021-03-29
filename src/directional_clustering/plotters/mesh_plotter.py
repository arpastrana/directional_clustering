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
        self.name = "Mesh Plus Plotter"

    def draw_vector_field(self, field, color, uniform, scale, width=0.5):
        """
        Draws a vector field on a mesh.

        It assumes the field and the mesh faces have the same keys.
        """
        _lines = []
        lines = []
        mesh = self.mesh

        for fkey in range(field.size()):
            vector = field[fkey]
            length = scale
            if not uniform:
                length = length_vector(vector) * scale
            _lines.append(line_sdl(mesh.face_centroid(fkey), vector, length))

        _lines = [line for line in map(line_tuple_to_dict, _lines)]

        for line in _lines:
            line["width"] = width
            line["color"] = color

        lines.extend(_lines)

        self.draw_lines(lines)
