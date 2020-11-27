from compas.datastructures import Mesh
from directional_clustering.fields import VectorField

__all__ = ["MeshPlus"]

class MeshPlus(Mesh):
    """
    Extand the original Mesh class with extra methods.
    """
    def get_vectorfield(self, name, vector_field = None):
        """
        Get or set the vector field under some certain attribute from or in a mesh.
        """
        if vector_field is None:
            vector_field = VectorField()
            for fkey in self.faces():
                vector_field.add_vector(fkey, self.face_attribute(fkey, name))
            return vector_field
        else:
            for vkey in vector_field.keys():
               self.face_attribute(vkey, name, vector_field.vector(vkey))


    def vectorfields(self):
        """
        Search for attributes of all supported vector fields in a mesh.
        """
        