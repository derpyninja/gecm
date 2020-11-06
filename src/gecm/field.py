import os
import numpy as np
import rasterio as rio
from rasterio.plot import show
import matplotlib.pyplot as plt
from src.gecm.dicts import int2class


class Map(object):
    """
    Implements playing field class.
    """

    def __init__(self, fpath, mapping):
        self.fpath = fpath
        self.src = rio.open(self.fpath)
        self.rows = self.src.width
        self.cols = self.src.height
        self.mapping = mapping

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
        return self.src.read(1, masked=masked)

    def convert(self):
        """
        Convert from integer to LULC representation.

        Returns
        -------
        np.ma
            Masked array of LULC strings
        """
        return int2class(self.read())

    def show(self):
        """
        Create a spatial plot of the map.

        Returns
        -------
        ax :
            matplotlib ax object
        """
        # read
        raster = self.read()

        # plot
        fig, ax = plt.subplots()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Map size: {} x {}".format(self.rows, self.cols))

        show(raster, ax=ax, transform=self.src.transform, cmap="Paired")
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
        bar_width = counts[:-1] / (self.rows * self.cols)

        # create barplot
        fig, ax = plt.subplots()
        ax.barh(y=int2class(unique[:-1]), width=bar_width)
        ax.set_xlabel("Percent of total area (%)")
        plt.tight_layout()
        return ax


if __name__ == "__main__":
    from src.gecm.dicts import nfi_mapping

    # define dirs
    data_raw = os.path.join("..", "..", "data", "raw")
    data_processed = os.path.join("..", "..", "data", "processed")
    figure_dir = os.path.join("..", "..", "plots")

    # load map
    fpath_map = os.path.join(data_processed, "NFI_rasterized_40_40.tif")
    field = Map(fpath=fpath_map, mapping=nfi_mapping)

    # create plot
    field.show()
    field.show_barh()

    # show plots
    plt.show()
