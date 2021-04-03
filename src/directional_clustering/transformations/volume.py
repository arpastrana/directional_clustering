from math import fabs

__all__ = ["volume_reinforcement_bending",
           "volume_reinforcement_bending_dir"]


def volume_reinforcement_bending(m1, m2, m12):
    """
    Estimates an upper bound for the reinforcement in a plate subjected to bending.
    """
    volume = fabs(m1) + fabs(m2) + 2 * fabs(m12)

    return volume


def volume_reinforcement_bending_dir(ma, mab):
    """
    Estimates an upper bound for the reinforcement in a plate on one direction.
    """
    # return fabs(ma) + fabs(mab)
    return max(ma + fabs(mab), -ma + fabs(mab))


if __name__ == "__main__":
    pass
