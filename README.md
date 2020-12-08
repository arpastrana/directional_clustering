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

## How to use this library?

After installation completed, we are able to play around with this library to realize 
a bunch of cool tasks related to directional clustering! We provide an example of how 
to utilize our library. Example scripts `01_clustering.py` and `01_plotting.py` are in 
folder `directional_clustering/scripts`.

`01_clustering.py` takes care of importing a mesh from JSON file, doing clustering on 
the choosen vector field and exporting the clustering results into another JSON file.

`01_plotting.py` takes care of resuming the previous clustering results from JSON file 
and visualizing the results.

Instuctions on how to run these two are as follows.

1. Go to the directory where scripts live (suppose you're currently at 
`directional_clustering`):

```bash
cd scripts
```

2.Run `01_clustering.py` with default command line arguments by typing the following.
In addition, you can pass customized values of command line arguments by `--flag` syntax
or taking them as positional arguments.
As usual, `--help` will lead you to our documentation for arguments.

```
python 01_clustering.py
```

3.There will be an inquriy about which vector filed do you want to do clustering on while 
`01_clustering.py` running (for exmaple you're choosing attribute `m_1`, input `m_1` will 
be passed to `vectorfield_tag`):

```bash
supported vector field attributes are:
 ['ps_2_top', 'ps_2_bot', 'ps_1_top', 'm_1', 'm_2', 'n_1', 'n_2', 'custom_2', 'ps_2_mid', 
 'ps_1_bot', 'ps_1_mid', 'custom_1']
please choose one attribute:m_1
```

4.Wait till the running process is over! The results will be automatically stored in a JSON
file named after `filename`_`vectorfield_tag`_`clustering_name`_`n_clusters`in folder 
`data/json_files`.

5. Run `01_plotting.py` with default command line arguments by typing the following.
In addition, you can pass customized values of command line arguments by `--flag` syntax
or taking them as positional arguments. 
As usual, `--help` will lead you to our documentation for arguments.

```
python 01_plotting.py
```

6.Plot will be shown as html.

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
