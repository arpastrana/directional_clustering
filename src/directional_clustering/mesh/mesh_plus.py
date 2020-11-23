from compas.datastructures import Mesh
from directional_clustering.fields import VectorField

__all__ = ["MeshPlus"]

class MeshPlus(Mesh):
    """
    Extand the original Mesh class with an extra method "get_vectorfield"
    """
    def get_vectorfield(self, tag):
        """
        Extracts a vector field from a mesh according to a tag.
        """
        vector_field = VectorField()

        for fkey in self.faces():
            vector_field.add_vector(fkey, self.face_attribute(fkey, tag))

        return vector_field