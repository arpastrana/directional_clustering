from directional_clustering.geometry import contour_polygons
from directional_clustering.geometry import line_tuple_to_dict
from directional_clustering.geometry import vector_lines_on_faces

from compas_plotters import MeshPlotter


__all__ = [
    "ClusterPlotter"
]

class ClusterPlotter(MeshPlotter):
    def __init__(self, *args, **kwargs):
        super(ClusterPlotter, self).__init__(*args, **kwargs)
        self.name = "Cluster Plotter"
    
    def draw_clusters_contours(self, centers, labels, density, method):
        mesh = self.mesh
        polygons = contour_polygons(mesh, centers, labels, density, method)
        self.draw_polylines(polygons)

    def draw_vector_field(self, tag, color, uniform, scale, width):
        mesh = self.mesh
        lines = []

        _lines = vector_lines_on_faces(mesh, tag, uniform, scale)
        _lines = [line for line in map(line_tuple_to_dict, _lines)]

        for line in _lines:
            line["width"] = width
            line["color"] = color

        lines.extend(_lines)

        self.draw_lines(lines)