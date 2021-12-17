import os

from compas.datastructures import Mesh

# extended version of Mesh
from directional_clustering.mesh import MeshPlus

# JSON file directory
from directional_clustering import JSON


# load a mesh from a JSON file
root = "clustered"
algo_name = "cosine_kmeans"
filename = "four_point_slab_k5_m_1_smooth0"
name_in = filename + ".json"
json_in = os.path.abspath(os.path.join(JSON, root, algo_name, name_in))
mesh = MeshPlus.from_json(json_in)

# remove unwanted face attributes
available_vf = mesh.vector_fields()
available_forces = ["mx", "my", "mxy", "nx", "ny", "nxy", "vx", "vy"]
preservable_vf = ["m_1", "m_2", "m_1_k", "m_2_k"]
removable_vf = set(available_vf + available_forces) - set(preservable_vf)

print(removable_vf)

for fkey, attr in mesh.facedata.items():
    for rvf in removable_vf:
        del attr[rvf]

print(mesh.vector_fields())

# export mesh
name_out = "{}_mark.json".format(filename)
json_out = os.path.abspath(os.path.join(JSON, root, algo_name, name_out))
mesh.to_json(json_out)
print("Exported clustered vector field with mesh to: {}".format(json_out))
