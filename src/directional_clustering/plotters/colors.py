from numpy import amin
from numpy import amax
from numpy import array

from compas.utilities import i_to_rgb
from compas.utilities import i_to_black
from compas.utilities import i_to_blue


__all__ = [
    "color_maker",
    "rgb_colors",
    "black_colors",
    "blue_colors"
]


def color_maker(data, callback, invert=False):

    assert isinstance(data, dict)

    dataarray = array(list(data.values()))
    valuemin = amin(dataarray)
    valuemax = amax(dataarray - valuemin)

    colors = {}
    for idx, value in data.items():
        centered_val = (value - valuemin)  # in case min is not 0
        if not invert:
            ratio = centered_val / valuemax  # tuple 0-255
        else:
            ratio = (valuemax - centered_val) / valuemax

        colors[idx] = callback(ratio)  # tuple 0-255

    return colors


def rgb_colors(data, invert=False):
    return color_maker(data, i_to_rgb, invert)


def black_colors(data, invert=False):
    return color_maker(data, i_to_black, invert)


def blue_colors(data, invert=True):
    return color_maker(data, i_to_blue, invert)


if __name__ == "__main__":
    pass
