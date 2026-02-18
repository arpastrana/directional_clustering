# os
import os

# argument parsing helper
import fire

# plots and beyond
import matplotlib.pyplot as plt

# time is running out
from datetime import datetime

# hello numpy, my old friend
import numpy as np

# Library file directories
from directional_clustering import DATA
from directional_clustering import JSON

# extended version of Mesh
from directional_clustering.mesh import MeshPlus

# transformations
from directional_clustering.transformations import smoothen_vector_field

# ==============================================================================
# Matplotlib beautification
# ==============================================================================

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rc('axes', linewidth=1.5)
plt.rc('axes', labelsize=15)
plt.rc('xtick', labelsize=20, direction="in")
plt.rc('ytick', labelsize=20, direction="in")
plt.rc('legend', fontsize=15)

# setting xtick parameters:
plt.rc('xtick.major', size=10, pad=4)
plt.rc('xtick.minor', size=5, pad=4)
plt.rc('ytick.major', size=10)
plt.rc('ytick.minor', size=5)

# ==============================================================================
# Convenience functions
# ==============================================================================

colors_studies = {"Clustered": "skyblue", "XY": "coral"}
case_studies_names = ["4PS", "PSS", "CCS"]

k_ratio = [1.14, 1.15, 1.18]
ortho_ratio = [1.25, 1.27, 1.47]

# ==============================================================================
# Main course
# ==============================================================================

def plot_bars(show_plot=False, save_img=True, figsize=(5, 5), dpi=600):
    """
    Plot the volume ratios to a PS field for the key 3 case studies.
    """

    plt.rcParams['hatch.linewidth'] = 0.5

    x = np.arange(len(case_studies_names))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    rects1 = ax.bar(x - width/2, k_ratio, width, label='Clustered', color=colors_studies["Clustered"], alpha=0.8)
    rects2 = ax.bar(x + width/2, ortho_ratio, width, label='XY', color=colors_studies["XY"], alpha=0.8)

    # patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')

    for thing in rects2:
        thing.set_hatch("\\")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylim(1.0, 1.5)

    ax.set_ylabel("Ratio to PS Baseline")
    ax.set_xlabel("Case Study")
    ax.set_title("Volume-Material Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(case_studies_names)
    ax.legend()

    plt.grid(ls="--", lw=0.5, which="major", axis="y")

    # save
    if save_img:
        dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        img_name = "volume_bars" + "_" + dt + ".png"
        img_path = os.path.abspath(os.path.join(DATA, "images", img_name))
        plt.tight_layout()
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0.1, dpi=600)
        print("Saved image to : {}".format(img_path))

    # show plot
    if show_plot:
        plt.show()

# ==============================================================================
# Executable
# ==============================================================================

if __name__ == '__main__':
    fire.Fire(plot_bars)
