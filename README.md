# Directional Clustering

Directional clustering of vector fields on meshes.

![Clustered stress field on a perimeter-supported slab](data/images/five_clusters.png)

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

## How to I use this thing?

**PLACEHOLDER. Please provide instructions on how to use the two driver codes.**.

## Installation

The simplest way to install `directional_clustering` is to build it from source
after cloning this repo. For developer mode, please jump to the next section.

1. First, we would need to install the latest version of
[Anaconda](https://www.continuum.io/). Anaconda will take care, among many other
things, of installing scientific computing packages like `numpy` and
`matplotlib` for us.

2. Next, let's create a new `conda` environment from your command line interface
(your terminal on macOS or from the anaconda prompt on windows).
The only required dependencies are `compas` and`sklearn`.

```bash
conda create -n clusters python=3.7 COMPAS=0.16.9 scikit-learn
conda activate clusters
```

3. We should clone `directional_clustering` from this repository and move inside.
If you are a macOS user and want to put it in your home folder:

```bash
cd ~
git clone https://github.com/arpastrana/directional_clustering.git
cd directional_clustering
```

4. Next, install `directional_clustering` as an editable package from source using `pip`:

```bash
pip install -e .
```

5. To double-check that everything is up and running, still in your command line
interface, let's type the following and hit enter:

```bash
python -c "import directional_clustering"
```

If no errors occur, smile 🙂! You have a working installation of
`directional_clustering`.

## Developer Mode

If you are rather interested in building the documentation, testing, or making a
pull request to this package, you should install this package slighly differently.

Concretely, instead of running `pip install -e .` in step 4 above, we must do:

```bash
pip install -r requirements-dev.txt
```

This will take care of installing additional dependencies like `sphinx` and `pytest`.

### Testing

To run the `pytest` suite automatically, type from the command line;

```bash
invoke test
```

### Documentation

To build this package's documentation in `html`, type:


```bash
invoke docs
```

You'll find the generated `html` data in the `docs/` folder.

If instead what we need is a manual in `pdf` format, let's run:


```bash
invoke pdf
```

The manual will be saved in `docs/latex` as `directional_clustering.pdf`.

## License

MIT
