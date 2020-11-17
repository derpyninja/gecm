import os
import numpy as np
import rasterio as rio
from rasterio.plot import show
import matplotlib.pyplot as plt
from src.gecm.functions import (
    invert_dict,
    convert_lulc_id_to_class,
    remap_lulc_dict,
    cmap2hex,
    remap_array_with_dict,
)


class Map(object):
    """
    Implements the main class of the game describing the playing field.
    """

    def __init__(
        self,
        fpath,
        original_lulc_mapping,
        lulc_remapping,
        simplified_lulc_mapping,
        cmap=None,
    ):
        """
        TODO
        Parameters
        ----------
        fpath
        original_lulc_mapping
        lulc_remapping
        simplified_lulc_mapping
        cmap : mpl.cm
            If None, is created dynamically based on granularity. If given,
            one has full manual control. This means, however, to manually take
            care of making the cm really fit with the underlying data.
        """
        self.fpath = fpath

        # rasterio src
        self.src = rio.open(self.fpath)

        # image dimensions
        self.rows = self.src.width
        self.cols = self.src.height

        # map raster data
        self.map_original = None
        self.map_simplified = None

        # LULC mappings
        self.original_lulc_mapping = original_lulc_mapping
        self.lulc_remapping = lulc_remapping
        self.simplified_lulc_mapping = simplified_lulc_mapping
        self.remapping_dict = None

        # colormap
        self.cmap = cmap
        self.cmap_str = "Paired"
        self.n_colors = None
        self.cmap_hex = None

    def initialise(self, masked=True, granularity=0):
        """
        Initialise map.

        Parameters
        ----------
        granularity
        masked : bool
            Whether to parse as np.ma

        Returns
        -------
        np.ma
            Parsed LULC map
        """
        self._read_lulc_data(masked=masked)

        # simplify
        if granularity == 1:
            self._simplify_lulc_data()

        print(self.cmap)
        # self._create_cmap(granularity=granularity)
        self.cmap_hex = cmap2hex(self.cmap)
        print(self.cmap_hex)

    def _read_lulc_data(self, masked=True):
        """
        Read map data.

        Parameters
        ----------
        masked : bool
            Whether to parse as np.ma

        Returns
        -------
        np.ma
            Parsed LULC map
        """
        self.map_original = self.src.read(1, masked=masked)
        return None

    def _simplify_lulc_data(self):
        """
        TODO

        Returns
        -------
        TODO
        """
        simplified_lulc_mapping_long = remap_lulc_dict(
            old_dict=self.original_lulc_mapping,
            remap_dict=self.lulc_remapping,
            remap_dict_ids=self.simplified_lulc_mapping,
        )

        # invert original mapping
        original_dict_inv = invert_dict(self.original_lulc_mapping)

        # create remapping scheme
        assert len(original_dict_inv.keys()) == len(
            simplified_lulc_mapping_long.values()
        )
        self.remapping_dict = dict(
            zip(original_dict_inv.keys(), simplified_lulc_mapping_long.values())
        )

        # simplify LULC representation by aggregating classes based on a
        # pre-defined remapping scheme
        self.map_simplified = remap_array_with_dict(
            input_array=self.map_original, mapping=self.remapping_dict
        )
        return None

    def _get_lulc_data_by_granularity(self, granularity=0):
        """
        TODO

        Parameters
        ----------
        granularity : int
            Integer in [0, 1], where 0 corresponds to the map with the original
            granularity and 1 to the simplified granularity.

        Returns
        -------
        (raster, dict_mapping)
        """
        if granularity == 0:
            return self.map_original, self.original_lulc_mapping
        elif granularity == 1:
            return self.map_simplified, self.remapping_dict
        else:
            raise NotImplementedError

    def _create_cmap(self, granularity=0):
        """
        TODO

        Parameters
        ----------
        granularity : int
            Integer in [0, 1], where 0 corresponds to the map with the original
            granularity and 1 to the simplified granularity.

        Returns
        -------
        TODO
        """
        if granularity == 0:
            raster = self.map_original
        elif granularity == 1:
            raster = self.map_simplified
        else:
            raise NotImplementedError

        self.n_colors = len(np.unique(raster)) - 1  # -1 for np.nan
        self.cmap = plt.get_cmap(self.cmap_str, lut=self.n_colors)
        self.cmap_hex = cmap2hex(self.cmap)
        return None

    def _convert(self):
        """
        Convert from integer to string representation.

        Returns
        -------
        np.ma
            Masked array of LULC strings
        """
        return convert_lulc_id_to_class(
            self.map_original, mapping=self.original_lulc_mapping
        )

    def update(self):
        # TODO
        pass

    def show(self, granularity=0):
        """
        Create a spatial plot of the map.

        Returns
        -------
        ax :
            matplotlib ax object
        """
        # define raster
        raster, mapping = self._get_lulc_data_by_granularity(
            granularity=granularity
        )

        # create figure
        fig, ax = plt.subplots()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Map size: {} x {}".format(self.rows, self.cols))
        print(self.cmap)
        show(raster, ax=ax, transform=self.src.transform, cmap=self.cmap)
        plt.tight_layout()
        return ax

    def show_bar(self, granularity=0):
        """
        Create a barplot of the current distribution of areal
        percentage for all LULC types.

        Returns
        -------
        ax :
            matplotlib ax object
        """
        # define raster
        raster, mapping = self._get_lulc_data_by_granularity(
            granularity=granularity
        )

        # get unique value counts
        unique, counts = np.unique(raster, return_counts=True)

        # bar width as percent of total pixels ~ areal percentage
        bar_width = np.array(counts[:-1] / (self.rows * self.cols))
        non_biosphere_area = 1 - bar_width.sum()

        # create barplot
        fig, ax = plt.subplots()
        classes = convert_lulc_id_to_class(
            int_array=unique[:-1], mapping=self.simplified_lulc_mapping
        )
        print(self.cmap_hex)
        ax.bar(x=classes, height=bar_width, color=self.cmap_hex)
        # ax.set_title(
        #    "Non-biosphere area: {:.2f} %".format(non_biosphere_area * 100)
        # )
        ax.set_xlabel("Percent of total area (%)")
        plt.tight_layout()
        return ax


if __name__ == "__main__":
    from src.gecm.dicts import (
        original_lulc_mapping,
        lulc_remapping,
        simplified_lulc_mapping,
        simplified_lulc_mapping_colors,
    )
    from matplotlib.colors import ListedColormap
    from pathlib import Path

    # define dirs
    abspath = os.path.abspath("")
    project_dir = str(Path(abspath).parents[1])
    data_raw = os.path.join(project_dir, "data", "raw")
    data_processed = os.path.join(project_dir, "data", "processed")
    figure_dir = os.path.join(project_dir, "plots")

    # playing field size
    rows = cols = 90

    # set path to raster data
    fpath_map = os.path.join(
        data_processed, "NFI_rasterized_{}_{}.tif".format(rows, cols)
    )

    # define cmap
    simplified_lulc_cm = ListedColormap(
        [
            simplified_lulc_mapping_colors[x]
            for x in simplified_lulc_mapping_colors.keys()
        ]
    )

    # load map
    field = Map(
        fpath=fpath_map,
        original_lulc_mapping=original_lulc_mapping,
        simplified_lulc_mapping=simplified_lulc_mapping,
        lulc_remapping=lulc_remapping,
        cmap=simplified_lulc_cm,
    )

    # initialise: this step is crucial, else nothing works!
    field.initialise(granularity=1)

    # plot
    field.show(granularity=1)
    field.show_bar(granularity=1)

    # show
    plt.show()
