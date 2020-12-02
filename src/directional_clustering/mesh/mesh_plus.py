from compas.datastructures import Mesh
from directional_clustering.fields import VectorField
<<<<<<< Updated upstream
=======
import os
>>>>>>> Stashed changes

__all__ = ["MeshPlus"]

class MeshPlus(Mesh):
    """
    Extand the original Mesh class with extra methods.
    """
<<<<<<< Updated upstream
    def get_vectorfield(self, name, vector_field = None):
=======
    def vectorfield(self, name, vector_field = None):
>>>>>>> Stashed changes
        """
        Get or set the vector field under some certain attribute from or in a mesh.
        """
        if vector_field is None:
            vector_field = VectorField()
            for fkey in self.faces():
                vector_field.add_vector(fkey, self.face_attribute(fkey, name))
            return vector_field
<<<<<<< Updated upstream
        else:
=======
        else: 
            assert vector_field.size() == self.number_of_faces(), "The vector field to add is incompatible with the mesh"
>>>>>>> Stashed changes
            for vkey in vector_field.keys():
               self.face_attribute(vkey, name, vector_field.vector(vkey))


<<<<<<< Updated upstream
    def vectorfields(self):
        """
        Search for attributes of all supported vector fields in a mesh.
        """
        
=======
    def all_vectorfields(self):
        """
        Search for attributes of all supported vector fields in a mesh.
        """
        
if __name__ == "__main__":
    from directional_clustering import JSON
    name_in = "perimeter_supported_slab.json"

    JSON_IN = os.path.abspath(os.path.join(JSON, name_in))
    mesh = Mesh.from_json(JSON_IN)
    
    #face = mesh.get_any_face()
    #print(face)
    attr2 = mesh.face_attributes(885, "m_1")
    print(attr2)
    

>>>>>>> Stashed changes
