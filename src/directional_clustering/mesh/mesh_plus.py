from compas.datastructures import Mesh
from directional_clustering.fields import VectorField
import os

__all__ = ["MeshPlus"]

class MeshPlus(Mesh):
    """
    Extand the original Mesh class with extra methods.
    """
    def vectorfield(self, name, vector_field = None):
        """
        Get or set the vector field under some certain attribute from or in a mesh.
        """
        if vector_field is None:
            vector_field = VectorField()
            for fkey in self.faces():
                try:
                    value = self.face_attribute(fkey, name)
                    if type(value) is list:
                        vector_field.add_vector(fkey, value)
                    else:
                        raise ValueError
                except ValueError:
                    return None        #the attribute doesn't exist or it's not a vectorfield
            return vector_field
        else: 
            assert vector_field.size() == self.number_of_faces(), "The vector field to add is incompatible with the mesh"
            for vkey in vector_field.keys():
                self.face_attribute(vkey, name, vector_field.vector(vkey))


    def all_vectorfields(self):
        """
        Search for attributes of all supported vector fields in a mesh.
        """
        fkey = self.get_any_face()

        attr_view = self.face_attributes(fkey)
        attr_default = list(attr_view.keys()) 
        attr_view.custom_only = True
        attr_costom = list(attr_view.keys())
        attr = attr_default + attr_costom

        attr_vectorfield = []
        for name in attr:
            if self.vectorfield(name) is not None: 
                attr_vectorfield.append(name)

        return attr_vectorfield



if __name__ == "__main__":
    from directional_clustering import JSON
    name_in = "perimeter_supported_slab.json"

    JSON_IN = os.path.abspath(os.path.join(JSON, name_in))
    mesh = MeshPlus.from_json(JSON_IN)

    attr = mesh.all_vectorfields()
    print(attr)

    fkey = mesh.get_any_face()
    for name in attr :
        value = mesh.face_attribute(fkey, name)
        print(value)
    

    

