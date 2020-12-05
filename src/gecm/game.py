import os
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.plot as rioplot
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap


from src.gecm import io, vis, base, dicts


class MatrixGame(object):
    """
    Implements the main class of the game.
    """

    def __init__(
        self,
        fpath,
        original_lulc_mapping,
        lulc_remapping,
        simplified_lulc_mapping,
        cmap=None,
        model_param_dict=None,
        model_calc_dict=None,
        config_file=None,
        credentials_fpath=None,
        df_mgmt_decisions_long=None,
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

        # currently, the code is only tested with quadratic maps
        assert self.src.width == self.src.height
        self.n_pixels = self.src.width
        self.rows = self.src.width
        self.cols = self.src.height

        # define management blocks (currently only works with n=4 blocks)
        self.n_blocks = 4
        self.n_pixels_per_block = int(self.n_pixels / self.n_blocks)
        self.block_definition_matrix_block_lvl = None  # n_blocks x n_blocks
        self.block_definition_matrix_pixel_lvl = None  # n_pixels x n_pixels
        self.unit_matrix = np.ones(shape=(self.rows, self.cols), dtype=np.uint8)

        # tourism and teamwork
        self.cooperation_matrix_block_lvl = np.full(
            (self.n_blocks, self.n_blocks), fill_value=False, dtype=bool
        )
        self.cooperation_matrix_pixel_lvl = np.full(
            (self.n_pixels, self.n_pixels), fill_value=False, dtype=bool
        )
        self.tourism_matrix_block_lvl = self.cooperation_matrix_block_lvl
        self.tourism_matrix_pixel_lvl = self.cooperation_matrix_pixel_lvl

        # raster data describing the map. one can only play on
        # the simplified lulc matrix playing field
        self.lulc_matrix_original = None
        self.lulc_matrix = None
        self.lulc_matrix_stack = None

        # LULC mappings
        self.original_lulc_mapping = original_lulc_mapping
        self.lulc_remapping = lulc_remapping
        self.simplified_lulc_mapping = simplified_lulc_mapping
        self.remapping_dict = None

        # management decision data
        self.df_mgmt_decisions_long = df_mgmt_decisions_long

        # property rights per block
        self.property_rights_matrix = None

        # Google APIs
        self.credentials_fpath = credentials_fpath

        # colormap
        # self.cmap = cmap
        self.cmap_str = "Paired"
        self.n_colors = None
        self.cmap_lulc_hex = None

        # map
        self.cmap_lulc = ListedColormap(
            [
                dicts.simplified_lulc_mapping_colors[x]
                for x in dicts.simplified_lulc_mapping_colors.keys()
            ]
        )

        # stakeholders
        self.cmap_stakeholders = ListedColormap(
            [
                dicts.stakeholder_color_dict[x]
                for x in dicts.stakeholder_color_dict.keys()
            ]
        )

        # configuration file data
        self.config_file = config_file
        self.model_param_dict = model_param_dict
        self.model_calc_dict = model_calc_dict

    def initialise(self, masked=True, granularity=1):
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
        # read raw data
        self._read_lulc_data(masked=masked)

        # define blocks in block-level matrix
        matrix_indizes = (
            np.indices((self.n_blocks, self.n_blocks), dtype="uint8") + 1
        )
        row_indizes, column_indizes = matrix_indizes[0], matrix_indizes[1]
        self.block_definition_matrix_block_lvl = np.char.add(
            row_indizes.astype(np.str), column_indizes.astype(np.str)
        ).astype(np.uint8)

        # define blocks in pixel-level matrix
        self.block_definition_matrix_pixel_lvl = np.multiply(
            self.unit_matrix,
            np.kron(
                self.block_definition_matrix_block_lvl,
                np.ones(
                    shape=(self.n_pixels_per_block, self.n_pixels_per_block)
                ),
            ),
        )

        # assign property rights based on hard-coded dict
        self._assign_property_rights()

        # optional: lulc map simplification (through spatial disaggregation)
        if granularity == 1:
            # simplify and store
            self._simplify_lulc_data()

            # create 3D array out of 2D array to store changed maps of all
            # rounds along the third (z-) axis dimension
            self.lulc_matrix_stack = self.lulc_matrix.reshape(
                (self.lulc_matrix.shape[0], self.lulc_matrix.shape[1], 1)
            )

        # self._create_cmap(granularity=granularity)
        self.cmap_lulc_hex = vis.cmap2hex(self.cmap_lulc)

    def fetch_mgmt_decisions(self):
        self.df_mgmt_decisions_long = io.parse_all_mgmt_decisions(
            config_file=self.config_file,
            credentials_fpath=self.credentials_fpath,
        )

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
        self.lulc_matrix_original = self.src.read(1, masked=masked)

    def _simplify_lulc_data(self):
        """
        TODO

        Returns
        -------
        TODO
        """
        simplified_lulc_mapping_long = base.remap_lulc_dict(
            old_dict=self.original_lulc_mapping,
            remap_dict=self.lulc_remapping,
            remap_dict_ids=self.simplified_lulc_mapping,
        )

        # invert original mapping
        original_dict_inv = base.invert_dict(self.original_lulc_mapping)

        # create remapping scheme
        assert len(original_dict_inv.keys()) == len(
            simplified_lulc_mapping_long.values()
        )
        self.remapping_dict = dict(
            zip(original_dict_inv.keys(), simplified_lulc_mapping_long.values())
        )

        # simplify LULC representation by aggregating classes based on a
        # pre-defined remapping scheme
        self.lulc_matrix = base.remap_array_with_dict(
            input_array=self.lulc_matrix_original, mapping=self.remapping_dict
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
            return self.lulc_matrix_original, self.original_lulc_mapping
        elif granularity == 1:
            return self.lulc_matrix, self.remapping_dict
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
            raster = self.lulc_matrix_original
        elif granularity == 1:
            raster = self.lulc_matrix
        else:
            raise NotImplementedError

        self.n_colors = len(np.unique(raster)) - 1  # -1 for np.nan
        self.cmap_lulc = plt.get_cmap(self.cmap_str, lut=self.n_colors)
        self.cmap_lulc_hex = vis.cmap2hex(self.cmap_lulc)
        return None

    def _convert(self):
        """
        Convert from integer to string representation.

        Returns
        -------
        np.ma
            Masked array of LULC strings
        """
        return base.convert_lulc_id_to_class(
            self.lulc_matrix_original, mapping=self.original_lulc_mapping
        )

    def _assign_property_rights(self):
        arr = self.block_definition_matrix_pixel_lvl.copy()
        for k, v in dicts.stakeholder_property_dict.items():
            for j in v:
                arr[arr == j] = k
        self.property_rights_matrix = arr

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
            cmap=self.cmap_lulc,
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
        Create a barplot of the current pixel distribution of all LULC types.

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

        classes = base.convert_lulc_id_to_class(
            int_array=unique[:-1], mapping=self.simplified_lulc_mapping
        )
        ax.bar(x=classes, height=bar_width, color=self.cmap_lulc_hex)
        # ax.set_xlabel("Percent of total area (%)")
        return ax

    def show_dashboard(
        self,
        granularity=1,
        figure_size=None,
        property_rights=False,
        relative=False,
    ):
        """
        Displays the game dashboard by wrapping the other plotting functions.

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
        # ---------------------------------------------------------------------

        # property rights
        if property_rights:
            rioplot.show(
                self.property_rights_matrix,
                ax=ax_map,
                transform=self.src.transform,
                cmap=self.cmap_stakeholders,
                contour=False,
                zorder=0,
                alpha=0.6,
            )

        # MAP
        self.show(granularity=granularity, ax=ax_map)

        # PROPERTY
        self.show_bar(granularity=granularity, relative=relative, ax=ax_bar)

        # GDP
        # TODO: create function based on the code below later on
        df_model_params = pd.DataFrame(self.model_param_dict, index=["value"]).T
        df_model_params_subset = df_model_params.loc[
            [
                "bank_account_farmer_1",
                "bank_account_farmer_2",
                "bank_account_forestry_1",
                "bank_account_forestry_1",
            ],
            :,
        ]
        df_model_params_subset["group"] = df_model_params_subset.index.values
        df_model_params_subset["group"] = df_model_params_subset["group"].apply(
            lambda x: x.split("_")[2]
        )
        df_model_params_subset = df_model_params_subset.reset_index()
        df_model_params_subset["index"] = df_model_params_subset["index"].apply(
            lambda x: "_".join(x.split("_")[2:4])
        )
        df_model_params_subset.groupby("group").sum().plot(
            ax=ax_gdp, kind="bar", stacked=True, legend=None, rot=0
        )

        # UNEMPLOYMENT
        # TODO [low]: implement based on martina's function and 'df_model_params_subset'

        # labeling
        ax_bar.set_ylabel("Property (# pixels)")
        ax_gdp.set_ylabel("GDP ($)")
        ax_gdp.set_xlabel(None)
        ax_empl.set_ylabel("Unempl. (%)")
        ax_empl.set_ylim(0, 100)

        # create title
        title = "Round {}".format(self.get_rounds(current=True))
        fig.suptitle(title)

        # tight
        gs.tight_layout(fig)
        # gs.update(top=0.95)

    def show_all_mgmt_decisions(self):
        return vis.show_all_mgmt_decisions(self.df_mgmt_decisions_long)


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
    field = MatrixGame(
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
    print(field.lulc_matrix.shape)

    # plot new map (and potentially also the changes)

    # show
    plt.show()
