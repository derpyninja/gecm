**G**ames in support of **E**cosystem **C**risis **M**anagement (gecm)
======================================================================

Welcome to our student project "**G**ames in support of **E**cosystem **C**risis **M**anagement". 
This project was created in Autumn 2020 as part of [Foundations of Ecosystem Management](https://ecology.ethz.ch/education/master-courses/foundations-of-ecosystem-management.html), 
a graduate-level course offered at **[ETH Zurich](https://ethz.ch/en.html)**. The team members were Lena Wunderlin, Martina Buck, Marco Barandun, 
Ella-Mona Chevalley and Felix Zaussinger.

![Example Dashboard](gecm_dashboard.png?raw=true "Dashboard")

Collaboration
=============

We tried to keep the core of the package rather flexible to leave ample room 
for the possibility of simultaneously addressing different application domains. 
However, even better abstraction of core functionalities would be an important 
prerequisite of enhanced flexibility. Ultimately, one could potentially 
play such games around the world and across ecosystem management problems: all
one needs is a sound conceptual model, and a GIS map to start with. If you share
the excitement for such an endeavour and would be interested in taking a lead on 
further developments of these (or other) aspects of the package, we would be 
happy and excited if you got in touch with Felix (*fzaussinger@ethz.ch*).

Implementation paradigm
=======================

The common, underlying data structure/interface consists of a set of matrices, 
which can be combined ("stacked") into a 3D array representation ("data cube"). 
The following 2D matrices of equal shape (currently only quadratic supported) 
need to be implemented to play a game. Note that all of them should be numpy 
masked arrays (np.ma), bto handle cropping out all *Land Use Land Cover* (LULC) data outside the 
area of interest (in this case, a biosphere park boundary).

This table summarizes **initial parameters** needed to describe the size and
blocks/plots of the **playing field**, given that a LULC raster map already
exists.

Parameter | Data structure | Data type | Shape | Description
--- | --- | --- | --- | ---
n_pixels | int | np.uint16 | (1,) | The size of the quadratic playing field along one dimension, where n_pixels = rows = cols.
n_blocks | int | np.uint8 | (1,) | The number of plots/blocks to be created.

From these initial parameters, more **parameters are derived**:

Parameter | Data structure | Data type | Shape | Description
--- | --- | --- | --- | ---
n_pixels_per_block | int | np.uint8 | (1,) | n_pixels_block := n_blocks / n_pixels.

This table summarizes all matrices necessary for fully describing the
initial **state of the playing field and for updating it after each round**:

Matrix | Variable name | Data structure | Data type | Shape | General description | Mapping description
--- | --- | --- | --- | --- | --- | ---
**Land-use and land-cover (LULC) types** | lulc_matrix | np.ma | np.unit8 | (n_pixels, n_pixels) | defines the land-cover and land-use types of the playing field | each integer maps to a unique LULC class.
**Plot/Block Definitions** | block_definition_matrix | np.ma | np.unit8 | (n_blocks, n_blocks) --> (n_pixels, n_pixels) |  defines the plots/blocks the players can manipulate | each integer maps to a unique plot/block identifier. this "small" matrix is brought into (n_pixels, n_pixels) shape via the kronecker delta function: (n_blocks, n_blocks) --> (n_pixels, n_pixels).
**Cooperation/Teamwork** | cooperation_matrix | np.ma | np.bool | (n_blocks, n_blocks) --> (n_pixels, n_pixels) |  defines in which block players from a certain stakeholder group are open for cooperation with players from other stakeholder groups. | TRUE for blocks/plots/pixels where stakeholders are open for cooperation, else FALSE. this "small" matrix is brought into (n_pixels, n_pixels) shape via the kronecker delta function: (n_blocks, n_blocks) --> (n_pixels, n_pixels).
**Tourism/SSDA** | tourism_matrix | np.ma | np.bool | (n_blocks, n_blocks) --> (n_pixels, n_pixels) |  defines the plot which SSDA designates as being particularly valuable for tourism based on biodiversity, etc. | TRUE for blocks/plots/pixels which the SSDA designated as particularly valuable for touristic activities, else FALSE. this "small" matrix is brought into (n_pixels, n_pixels) shape via the kronecker delta function: (n_blocks, n_blocks) --> (n_pixels, n_pixels).

Note
====
This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.