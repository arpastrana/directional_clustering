# Directional Clustering

Directional clustering of vector fields on meshes.

## Introduction

The initial motivation of this work revolved around principal stress fields.
In principle, they suggest good directions to orient material efficiently in
architectural structures. This implies that by following these directions,
less material would be used to achieve a target level of structural performance.

Principal stress fields are ubiquitously computed by off-the-shelf FEA software
and are represented as a cloud of vectors (i.e. a vector field).

As principal stress fields are heterogeneous and form continuous curvilinear
trajectories, it is actually difficult for fabrication reasons to place material
(in the form reinforcement bars or beams) in a way that exactly match the field
directions. It is almost cumbersome, and this is probably one of the reasons why
we actually keep on building with orthogonal grids everywhere (take a look at
the room around you, for example).

In this work we question the heterogeneity of a principal stress field and
inquiry on how much we can simplify it so that we can maximize fabricability
while compromising as little as possible in structural performance. In short,
what we want is to find the lowest possible amount of different vectors that
encode the maximum amount of directional information about a principal stress
field. We leverage clustering methods to this end.

Zooming out a bit, this work can be extended to consider other non-structural
vector fields such as curvature, or combinations thereof.

This work was initiated by Rafael Pastrana in the School of Architecture at
Princeton University in 2020.

## Getting Started

Check out the introductory example in the [scripts folder](https://github.com/arpastrana/directional_clustering/blob/apc524/scripts/01_directional_clustering.py).

## Installation

The best way to install `directional_clustering` is to build it from source
after cloning this repo.

First, we would need to install the latest version of
[Anaconda](https://www.continuum.io/).

Next, create a new `conda` environment from your command line interface
(your terminal on macOS or from the anaconda prompt on windows).
The only required dependencies are `python` version and `compas`.

```bash
conda create -n clusters python=3.7 COMPAS=0.16.9
conda activate clusters
```

We need to clone `directional_clustering` from this repository.
If you are a macOS user and want to put it in your home folder:

```bash
cd ~
git clone https://github.com/arpastrana/directional_clustering.git
```

Next, Move into the the repository's folder (the one you've just cloned) and
install `directional_clustering` as an editable package from source using `pip`:

```bash
cd directional_clustering
pip install -e .
```

To double-check that everything is up and running, still in your command line
interface, type the following and hit enter:

```bash
python -c "import directional_clustering"
```

If no errors occur, smile 🙂! You have a working installation of
`directional_clustering`.

## Contributing

Pull requests are welcome!

Make sure to read the [contribution
guide](https://github.com/arpastrana/directional_clustering/tree/master/CONTRIBUTING.md).
Please don't forget to run ``invoke test`` in your command line before making a
pull request.

## Issue tracker

If you find a bug or want to suggest a potential enhancement,
please help us tackling it by [filing a
report](https://github.com/arpastrana/directional_clustering/issues).

## License

MIT.
