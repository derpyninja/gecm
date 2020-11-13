import os
import numpy as np
import rasterio as rio
from rasterio.plot import show
import matplotlib as mpl
import matplotlib.pyplot as plt
from src.gecm.dicts import int2class, remap_lulc_dict


def cmap2hex(cm):
    """
    Extract color array from colormap object.

    Parameters
    ----------
    cm : mpl.colormap

    Returns
    -------
    list :
        List of colors in HEX format.
    """
    return [mpl.colors.rgb2hex(cm(i)[:3]) for i in range(cm.N)]


class Map(object):
    """
    Implements playing field class.
    """

    def __init__(self, fpath, original_nfi_mapping, remap_dict, remap_dict_ids, cmap="Paired"):
        self.fpath = fpath
        self.src = rio.open(self.fpath)
        self.rows = self.src.width
        self.cols = self.src.height

        self.original_nfi_mapping = original_nfi_mapping
        self.remap_dict = remap_dict
        self.remap_dict_ids = remap_dict_ids

        self.simplified_nfi_mapping = remap_lulc_dict(
            old_dict=self.original_nfi_mapping,
            remap_dict=self.remap_dict,
            remap_dict_ids=self.remap_dict_ids
        )

        self.cmap_str = cmap

        # to be set later
        self.field_detailed = None
        self.field_simplified = None
        self.n_colors = None
        self.cmap = None
        self.cmap_hex = None

    def read(self, masked=True):
        """
        Read geotiff via rasterio.

        Parameters
        ----------
        masked : bool
            Whether to parse as np.ma

        Returns
        -------
        np.ma
            Parsed LULC map
        """
        field_array = self.src.read(1, masked=masked)

        # update
        self.field_detailed = field_array
        #self.field_simplified = self.field_detailed.apply()
        self.n_colors = len(np.unique(self.field_detailed)) - 1  # -1 for np.nan
        self.cmap = plt.get_cmap(self.cmap_str, lut=self.n_colors)
        self.cmap_hex = cmap2hex(self.cmap)

        return field_array

    def simplify(self):
        """Simplify map based on remapping dict."""
        pass

    def update(self):
        """Update map."""
        pass

    def convert(self):
        """
        Convert from integer to string representation.

        Returns
        -------
        np.ma
            Masked array of LULC strings
        """
        return int2class(self.field_detailed, mapping=self.original_nfi_mapping)

    def show(self):
        """
        Create a spatial plot of the map.

        Returns
        -------
        ax :
            matplotlib ax object
        """
        fig, ax = plt.subplots()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Map size: {} x {}".format(self.rows, self.cols))

        show(
            self.field_detailed, ax=ax, transform=self.src.transform,
            cmap=self.cmap
        )
        plt.tight_layout()
        return ax

    def show_barh(self):
        """
        Create a barplot of the current distribution of areal
        percentage for all LULC types.

        Returns
        -------
        ax :
            matplotlib ax object
        """
        raster = self.read()

        # get unique value counts
        unique, counts = np.unique(raster, return_counts=True)

        # bar width as percent of total pixels ~ areal percentage
        bar_width = np.array(counts[:-1] / (self.rows * self.cols))
        non_biosphere_area = 1 - bar_width.sum()

        # create barplot
        fig, ax = plt.subplots()
        classes = int2class(int_array=unique[:-1], mapping=self.original_nfi_mapping)
        ax.barh(y=classes, width=bar_width, color=self.cmap_hex)
        ax.set_title(
            "Non-biosphere area: {:.2f} %".format(
                non_biosphere_area * 100)
        )
        ax.set_xlabel("Percent of total area (%)")
        plt.tight_layout()
        return ax


if __name__ == "__main__":
    from src.gecm.dicts import nfi_mapping, nfi_mapping_v2, id_mapping

    # define dirs
    data_raw = os.path.join("..", "..", "data", "raw")
    data_processed = os.path.join("..", "..", "data", "processed")
    figure_dir = os.path.join("..", "..", "plots")

    # size
    rows = cols = 90

    # load map
    fpath_map = os.path.join(data_processed, "NFI_rasterized_{}_{}.tif".format(rows, cols))
    field = Map(
        fpath=fpath_map,
        original_nfi_mapping=nfi_mapping,
        remap_dict=nfi_mapping_v2,
        remap_dict_ids=id_mapping
    )

    # read (!)
    field.read()

    # TODO: simplify

    # plot
    #field.show()
    #field.show_barh()
    #plt.show()
