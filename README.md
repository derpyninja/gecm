**G**ames in support of **E**cosystem **C**risis **M**anagement (gecm)
======================================================================

Welcome to our student project on **G**ames in support of **E**cosystem **C**risis **M**anagement. 
This repository was created as part of [Foundations of Ecosystem Management](https://ecology.ethz.ch/education/master-courses/foundations-of-ecosystem-management.html), 
a graduate-level course offered at **ETH Zurich** in the autumn semester of 2020. 
The team members are Martina, Marco, Lena, Ella-Mona and Felix.

Implementation paradigm
=======================

Let's now define the common, underlying data structure/interface needed to
play our game. It consists of a set of matrices, which in turn can (but don't
necessarily need) be combined ("stacked") into a 3D array representation ("data cube").

To be flexible, I would suggest we stick with 2D arrays for now as they are
easier to change/adjust/adapt and less prone to errors. I would only transition
towards a 3D representation later on, given there is time and motivation to do so.

We probably need the following 2D matrices/arrays of equal shape (same number of
rows and columns, always!!) to be able to play our game. Note that all of them
should be numpy masked arrays (np.ma), because we crop out all LULC data
outside of our area of interest (the biosphere park boundary).

This table summarizes ** initial parameters needed to describe the size and
blocks/plots of the playing field** (given that a LULC raster map already
exists, which it does in our case).

Parameter | Data structure | Data type | Shape | Description
--- | --- | --- | --- | ---
n_pixels | int | np.uint16 | (1,) | The size of the quadratic playing field along one dimension, where n_pixels = rows = cols.
n_blocks | int | np.uint8 | (1,) | The number of plots/blocks to be created.

From these initial parameters, we create more **"derived parameters"**:

Parameter | Data structure | Data type | Shape | Description
--- | --- | --- | --- | ---
n_pixels_block | int | np.uint8 | (1,) | n_pixels_block := n_blocks / n_pixels.


This table summarizes **all matrices necessary for fully describing the
initial state of the playing field AND updating it after each round**:

Matrix | Variable name | Data structure | Data type | Shape | General description | Mapping description
--- | --- | --- | --- | --- | --- | ---
**Land-use and land-cover (LULC) types** | lulc_matrix | np.ma | np.unit8 | (n_pixels, n_pixels) | defines the land-cover and land-use types of the playing field | each integer maps to a unique LULC class.
**Plot/Block Definitions** | plot_definition_matrix | np.ma | np.unit8 | (n_blocks, n_blocks) --> (n_pixels, n_pixels) |  defines the plots/blocks the players can manipulate | each integer maps to a unique plot/block identifier. this "small" matrix is brought into (n_pixels, n_pixels) shape via the kronecker delta function: (n_blocks, n_blocks) --> (n_pixels, n_pixels).
**Cooperation/Teamwork** | cooperation_matrix | np.ma | np.bool | (n_blocks, n_blocks) --> (n_pixels, n_pixels) |  defines in which block players from a certain stakeholder group are open for cooperation with players from other stakeholder groups. | TRUE for blocks/plots/pixels where stakeholders are open for cooperation, else FALSE. this "small" matrix is brought into (n_pixels, n_pixels) shape via the kronecker delta function: (n_blocks, n_blocks) --> (n_pixels, n_pixels).
**Tourism/SSDA** | tourism_matrix | np.ma | np.bool | (n_blocks, n_blocks) --> (n_pixels, n_pixels) |  defines the plot which SSDA designates as being particularly valuable for tourism based on biodiversity, etc. | TRUE for blocks/plots/pixels which the SSDA designated as particularly valuable for touristic activities, else FALSE. this "small" matrix is brought into (n_pixels, n_pixels) shape via the kronecker delta function: (n_blocks, n_blocks) --> (n_pixels, n_pixels).

Collaboration
=============

If you want to co-develop the package, please follow the steps outlined [here](https://pypi.org/project/PyScaffold). 
In addition, [this blogpost](https://florianwilhelm.info/2018/11/working_efficiently_with_jupyter_lab/)
is an excellent entry point for learning how to organise your code well, especially
in view of using *jupyter notebooks* within *jupyter lab*.

Note
====

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
