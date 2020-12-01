import os
import numpy as np
import rasterio as rio
import rasterio.plot as rioplot
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from src.gecm.vis import cmap2hex
from src.gecm.base import (
    invert_dict,
    convert_lulc_id_to_class,
    remap_lulc_dict,
    remap_array_with_dict,
)


class Map(object):
    """
    Implements the main class of the game.
    """
    # TODO [low]: rename class to "ConceptualModel"
    def __init__(
        self,
        fpath,
        original_lulc_mapping,
        lulc_remapping,
        simplified_lulc_mapping,
        cmap=None,
        model_param_dict=None,
        model_calc_dict=None
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

        # initialise the zeroth' round of the game
        self.current_round = 0
        self.rounds_played = [self.current_round]

        # parse rasterio data & metadata
        self.src = rio.open(self.fpath)
        self.bbox = self.src.bounds
        self.rows = self.src.width
        self.cols = self.src.height

        # raster data describing the map
        self.map_original = None
        self.map_simplified = None
        self.map_simplified_container = {}

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

        # conceptual model (!)
        self.model_param_dict = model_param_dict
        self.model_calc_dict = model_calc_dict

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

        if granularity == 1:
            # simplify and store
            self._simplify_lulc_data()

            # append to empty container storing simplified maps of all rounds
            self.map_simplified_container[
                "round_{}".format(self.current_round)
            ] = self.map_simplified

        # self._create_cmap(granularity=granularity)
        self.cmap_hex = cmap2hex(self.cmap)

    def get_rounds(self, current=True):
        """

        Parameters
        ----------
        current

        Returns
        -------

        """
        round_idx = self.current_round

        if current:
            return self.rounds_played[round_idx]
        else:
            return self.rounds_played

    def update_round_number(self):
        """

        Returns
        -------

        """
        self.current_round += 1
        self.rounds_played.append(self.current_round)

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

    def show(self, granularity=1, figure_size=None, ax=None):
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
        if ax is None:
            _, ax = plt.subplots(figsize=figure_size)

        # show
        rioplot.show(
            raster,
            ax=ax,
            transform=self.src.transform,
            cmap=self.cmap,
            contour=False,
        )

        # obtain limits
        lower_left_x, lower_left_y, upper_right_x, upper_right_y = self.bbox

        # calculate total distance covered in x and y directions
        x_distance = upper_right_x - lower_left_x
        y_distance = upper_right_y - lower_left_y

        # get maximum of both directions and round through ceiling
        max_distance_ceiled = np.around(
            np.max([x_distance, y_distance]), decimals=-2
        )

        # set axis limits
        ax.set_ylim(lower_left_y, upper_right_y)
        ax.set_xlim(lower_left_x, upper_right_x)

        # grid params
        n_gridlines = 4
        spacing = max_distance_ceiled / n_gridlines
        lowest_line = lower_left_y + spacing
        leftest_line = lower_left_x + spacing

        # define grid properties
        grid_line_style = "--"
        grid_line_style_quadrants = "-"
        grid_line_width = 1.5
        grid_line_width_quadrants = 2
        grid_line_color = "grey"
        grid_line_color_quadrants = "black"

        # init containers for ticks and tick-labels
        x_ticks = []
        y_ticks = []
        x_tick_labels = np.arange(1, n_gridlines + 1, 1)
        y_tick_labels = np.arange(1, n_gridlines + 1, 1)

        # build grid
        for i, step in enumerate(np.arange(0, n_gridlines)):
            # calc and append y steps
            y_step = lowest_line + spacing * step
            y_ticks.append(y_step)

            # calc and append x steps
            x_step = leftest_line + spacing * step
            x_ticks.append(x_step)

            # draw y grid lines
            ax.axhline(
                y=y_step,
                linestyle=grid_line_style,
                linewidth=grid_line_width,
                color=grid_line_color,
            )

            # draw x grid lines
            ax.axvline(
                x=x_step,
                linestyle=grid_line_style,
                linewidth=grid_line_width,
                color=grid_line_color,
            )

            # draw block grid lines
            if i == 1:
                # draw y grid lines
                ax.axhline(
                    y=y_step,
                    linestyle=grid_line_style_quadrants,
                    linewidth=grid_line_width_quadrants,
                    color=grid_line_color_quadrants,
                )

                # draw x grid lines
                ax.axvline(
                    x=x_step,
                    linestyle=grid_line_style_quadrants,
                    linewidth=grid_line_width_quadrants,
                    color=grid_line_color_quadrants,
                )

        # shift ticks to midpoints of blocks
        x_ticks_shifted = np.array(x_ticks) - (spacing * 0.5)
        y_ticks_shifted = np.array(y_ticks) - (spacing * 0.5)

        # set ticks
        ax.set_xticks(ticks=x_ticks_shifted, minor=False)
        ax.set_yticks(ticks=y_ticks_shifted, minor=False)

        # set tick labels
        ax.set_xticklabels(labels=x_tick_labels, minor=False)
        ax.set_yticklabels(labels=y_tick_labels[::-1], minor=False)

        return ax

    def show_bar(
        self, granularity=1, relative=False, figure_size=None, ax=None
    ):
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
        if relative:
            bar_width = np.array(counts[:-1] / (self.rows * self.cols))
        else:
            bar_width = np.array(counts[:-1])

        # create barplot
        if ax is None:
            _, ax = plt.subplots(figsize=figure_size)

        classes = convert_lulc_id_to_class(
            int_array=unique[:-1], mapping=self.simplified_lulc_mapping
        )
        ax.bar(x=classes, height=bar_width, color=self.cmap_hex)
        # ax.set_xlabel("Percent of total area (%)")
        return ax

    def show_dashboard(self, granularity=1, figure_size=None, relative=False):
        """
        Displays the game dashboard by wrapping around the other plot functions.

        Parameters
        ----------
        layout
        granularity
        figure_size

        Returns
        -------

        """
        # layout
        nrows, ncols = 4, 5

        # create figure
        fig = plt.figure(constrained_layout=False, figsize=figure_size)
        gs = GridSpec(nrows, ncols, figure=fig)

        # create axes
        ax_map = plt.subplot(gs.new_subplotspec((0, 0), colspan=3, rowspan=4))
        ax_bar = plt.subplot(gs.new_subplotspec((0, 3), colspan=2, rowspan=2))
        ax_gdp = plt.subplot(gs.new_subplotspec((2, 3), colspan=2, rowspan=1))
        ax_empl = plt.subplot(gs.new_subplotspec((3, 3), colspan=2, rowspan=1))

        # populate axes with plots
        self.show(granularity=granularity, ax=ax_map)
        self.show_bar(granularity=granularity, relative=relative, ax=ax_bar)

        # temporary placeholders
        # TODO [minor]: add plots using real data once ready
        ax_gdp.text(0.5, 0.5, "GDP", va="center", ha="center")
        ax_empl.text(0.5, 0.5, "Unemployment", va="center", ha="center")

        # create title
        title = "Round {}".format(self.get_rounds(current=True))
        fig.suptitle(title)

        # tight
        gs.tight_layout(fig)
        # gs.update(top=0.95)


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
    rows = cols = 40

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

    # specify granularity
    gran = 1

    # initialise: this step is crucial, else nothing works!
    field.initialise(granularity=gran)
    print(field.current_round)

    # plot
    # field.show(granularity=granularity)
    # field.show_bar(granularity=granularity)

    # update round
    field.update_round_number()

    # some changes
    print(field.map_simplified.shape)

    # plot new map (and potentially also the changes)

    # show
    plt.show()
