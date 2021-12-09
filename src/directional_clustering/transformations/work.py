from directional_clustering.transformations import rigidity_bending_plate
from directional_clustering.transformations import curvatures_bending_plate


__all__ = ["virtual_work_bending"]


def virtual_work_bending(m1, m2, m12, height, e_modulus, poisson):
    """
    Computes the virtual work density from bending on a tiny fragment of a plate.
    """
    D = rigidity_bending_plate(height, e_modulus, poisson)
    k1, k2, k12 = curvatures_bending_plate(m1, m2, m12, D, poisson)

    work = m1 * k1 + m2 * k2 + 2 * m12 * k12

    return work


if __name__ == "__main__":
    pass
