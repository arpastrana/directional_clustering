********************************************************************************
Getting Started
********************************************************************************

.. _Anaconda: https://www.continuum.io/

.. highlight:: bash


Installation
============

The simplest way to install ``directional_clustering`` is to build it from source
after cloning this repo. For developer mode, please jump to the next section.

1. First, we would need to install the latest version of
`Anaconda`_. Anaconda will take care, among many other
things, of installing scientific computing packages like ``numpy`` and
``matplotlib`` for us.

2. Next, let's create a new ``conda`` environment from your command line interface
(your terminal on macOS or from the anaconda prompt on windows).
The only required dependencies are ``compas`` and ``sklearn``.

::

   conda create -n clusters python=3.7 COMPAS=1.8.1 scikit-learn
   conda activate clusters

3. We should clone ``directional_clustering`` from this repository and move inside.
If you are a macOS user and want to put it in your home folder:

::

   cd ~
   git clone https://github.com/arpastrana/directional_clustering.git
   cd directional_clustering

4. Next, install ``directional_clustering`` as an editable package from source using ``pip``:

::

   pip install -e .


5. To double-check that everything is up and running, still in your command line
interface, let's type the following and hit enter:

::

   python -c "import directional_clustering"


If no errors occur, smile :)! You have a working installation of
``directional_clustering``.

Developer Mode
==============

If you are rather interested in building the documentation, testing, or making a
pull request to this package, you should install this package slighly differently.

Concretely, instead of running ``pip install -e .`` in step 4 above, we must do:

::

   pip install -r requirements-dev.txt

This will take care of installing additional dependencies like ``sphinx`` and ``pytest``.

Testing
-------

To run the ``pytest`` suite automatically, type from the command line;

::

   invoke test

Documentation
-------------

To build this package's documentation in ``html``, type:

::

   invoke docs

You'll find the generated ``html`` data in the ``docs/`` folder.

If instead what we need is a manual in ``pdf`` format, let's run:

::

   invoke pdf

The manual will be saved in ``docs/latex`` as ``directional_clustering.pdf``.
