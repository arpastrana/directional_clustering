import os

from compas.datastructures import Mesh

from directional_clustering.fields import VectorField


__all__ = ["MeshPlus"]


class MeshPlus(Mesh):
    """
    An extended COMPAS mesh with specialized methods to parse vector fields.

    Parameters
    -----------
    args : `list`, optional.
        Default arguments.
    kwargs : `dict`, optional.
        Default keyword arguments.

    Notes
    -----
    See `help(compas.datastructures.Mesh)` for details on the constructor's signature.
    """
    def vector_field(self, name, vector_field=None):
        """
        Gets or sets a vector field that lives on the mesh.

        Parameters
        -----------
        name  : `str`
            The name of the vector field to get or to set.
        vector_field  : `directional_clustering.fields.VectorField`, optional.
            The vector field to store. Defaults to `None`.

        Returns
        --------
        vector_field : `directional_clustering.fields.VectorField`.
            The fetched vector field if a `name` was input.

        Notes
        -----
        Vector fields are stored a face attributes of a mesh.
        Refer to `compas.datastructures.face_attribute()` for more details.
        """
        if vector_field is None:
            vector_field = VectorField()
            for fkey in self.faces():
                try:
                    vector = self.face_attribute(fkey, name)
                    if type(vector) is list:
                        vector_field.add_vector(fkey, vector)
                    else:
                        raise ValueError
                except ValueError:
                    return None        #the attribute doesn't exist or it's not a vectorfield
            return vector_field
        else:
            msg = "The vector field to add is incompatible with the mesh"
            assert vector_field.size() == self.number_of_faces(), msg
            for vkey in vector_field.keys():
                self.face_attribute(vkey, name, vector_field.vector(vkey))


    def vector_fields(self):
        """
        Queries the names of all the vector fields stored on the mesh.

        Returns
        -------
        attr_vectorfield : `list`
            A list of with the vector field names.
        """
        fkey = self.get_any_face()

        attr_view = self.face_attributes(fkey)
        attr_default = list(attr_view.keys())
        attr_view.custom_only = True
        attr_costom = list(attr_view.keys())
        attr = attr_default + attr_costom

        attr_vectorfield = []
        for name in attr:
            if self.vector_field(name) is not None:
                attr_vectorfield.append(name)

        return attr_vectorfield


    def clustering_label(self, name, labels=None):
        """
        Gets or sets cluster labels on a mesh.

        Parameters
        -----------
        name  : `str`
            The name of the cluster label.
            The format is {vector_field_name}_{algorithm}_{number_of_clusters}.
        labels  : `dict`, optional.
            The cluster labels to store. Defaults to `None`.

        Returns
        --------
        labels  : `dict`
            The fetched labels only if a `name` was input.
        """
        if labels is None:
            labels = {}
            for fkey in self.faces():
                labels[fkey] = self.face_attribute(fkey, name)
            return labels
        else:
            for fkey in self.faces():
                label = labels[fkey]
                self.face_attribute(fkey, name, label)


if __name__ == "__main__":
    pass
