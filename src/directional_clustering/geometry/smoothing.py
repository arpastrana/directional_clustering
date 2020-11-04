from numpy import array
from numpy import mean


__all__ = [
    "laplacian_smoothed"
]


def laplacian_smoothed(mesh, data, iters, damping=0.5):
    
    data = {k: v for k, v in data.items()}

    for i in range(iters):

        iter_smoothed = {}

        for fkey in mesh.faces():
            f_data = data[fkey]
            nbr_data = array([data[key] for key in mesh.face_neighbors(fkey)])
            nbr_data = mean(nbr_data, axis=0)
            iter_smoothed[fkey] = f_data + (1 - damping) * (nbr_data - f_data)

        for fkey in mesh.faces():
            data[fkey] = iter_smoothed[fkey]

    return data


if __name__ == "__main__":
    pass
