********************************************************************************
Introduction
********************************************************************************

.. rst-class:: lead

Directional clustering of vector fields on meshes.

.. figure:: ../data/images/five_clusters.png
    :figclass: figure
    :class: figure-img img-fluid

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
