__all__ = ["strain_energy_bending",
           "rigidity_bending_plate",
           "curvatures_bending_plate"]


def strain_energy_bending(m1, m2, m12, height, e_modulus, poisson):
    """
    Computes the bending strain energy on a tiny fragment of a plate.
    """
    D = rigidity_bending_plate(height, e_modulus, poisson)
    k1, k2, k12 = curvatures_bending_plate(m1, m2, m12, D, poisson)

    energy = (k1 + k2) ** 2 - 2 * (1 - poisson) * (k1 * k2 - k12 ** 2)

    return 0.5 * D * energy


def rigidity_bending_plate(height, e_modulus, poisson):
    """
    Calculates the bending rigidity of a plate.
    """
    return e_modulus * (height ** 3) / (12 * (1 - poisson ** 2))


def curvatures_bending_plate(m1, m2, m12, D, poisson):
    """
    Calculates the curvature values on a plate caused by bi-directional bending.
    """
    k1 = (1 / (D * (1 - poisson ** 2))) * (m1 - poisson * m2)
    k2 = (1 / (D * (1 - poisson ** 2))) * (m2 - poisson * m1)
    k12 = (1 / (D * (1 - poisson))) * m12

    return k1, k2, k12


if __name__ == "__main__":
    pass
