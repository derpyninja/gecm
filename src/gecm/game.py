import os
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.plot as rioplot
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from src.gecm import io, model, vis, base, dicts


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

        # game choices
        # TODO: move to config
        self.brexit_round = 4
        self.teamwork_round = 1
        self.players = list(dicts.stakeholder_id_dict.keys())

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

        # map w/o cattle on the field
        self.cmap_lulc_no_cattle = ListedColormap(
            [
                dicts.simplified_lulc_mapping_colors[x]
                for x in list(dicts.simplified_lulc_mapping_colors.keys())[:-1]
            ]
        )

        # map with cattle on the field
        self.cmap_lulc = ListedColormap(
            [
                dicts.simplified_lulc_mapping_colors[x]
                for x in list(dicts.simplified_lulc_mapping_colors.keys())
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

        # initialise main conceptual model data store as a nested dictionary
        # ---------------------------------------------------------------------
        self.data_store = {
            # variable model parameters (changing per round)
            "variable_model_params": {
                "teamwork": [
                    False
                ],  # True/False for each game round, defaults to False for rounds 0, 1 and 2
                "area_sheep_total": [],
                "area_n_forest_total": [],
                "area_c_forest_total": [],
                "area_cattle_total": [0],
                "sheep_price": [self.model_param_dict["income_farmland_sheep"]],
                "cattle_price": [
                    self.model_param_dict["income_farmland_cattle"]
                ],
                "c_forest_price": [
                    self.model_param_dict["income_forest_native"]
                ],
                "n_forest_price": [
                    self.model_param_dict["income_forest_commercial"]
                ],
            },
            "Farmer_1": {
                # to be updated after calculating money_farmer and money_forester (!)
                "income": [self.model_param_dict["gdp_average"]],
                # to be updated after calculating money_farmer and money_forester (!)
                "bank_account": [
                    self.model_param_dict["bank_account_farmer_1"]
                ],
                # to be updated after calculating unemployment_rate after money_farmer and money_forester (!)
                "unemployment": [self.model_param_dict["unempl_rate_scotland"]],
                "area_sheep": [],  # to be updated a bit later
                "area_cattle": [0],  # to be updated a bit later
                "area_c_forest": [],  # to be updated a bit later
                "area_n_forest": [],  # to be updated a bit later
            },
            "Farmer_2": {
                # to be updated after calculating money_farmer and money_forester (!)
                "income": [self.model_param_dict["gdp_average"]],
                # to be updated after calculating money_farmer and money_forester (!)
                "bank_account": [
                    self.model_param_dict["bank_account_farmer_2"]
                ],
                # to be updated after calculating unemployment_rate after money_farmer and money_forester (!)
                "unemployment": [self.model_param_dict["unempl_rate_scotland"]],
                "area_sheep": [],  # to be updated a bit later
                "area_cattle": [0],  # to be updated a bit later
                "area_c_forest": [],  # to be updated a bit later
                "area_n_forest": [],  # to be updated a bit later
            },
            "Forester_1": {
                # to be updated after calculating money_farmer and money_forester (!)
                "income": [self.model_param_dict["gdp_average"]],
                # to be updated after calculating money_farmer and money_forester (!)
                "bank_account": [
                    self.model_param_dict["bank_account_forestry_1"]
                ],
                # to be updated after calculating unemployment_rate after money_farmer and money_forester (!)
                "unemployment": [self.model_param_dict["unempl_rate_scotland"]],
                "area_sheep": [],  # to be updated a bit later
                "area_cattle": [0],  # to be updated a bit later
                "area_c_forest": [],  # to be updated a bit later
                "area_n_forest": [],  # to be updated a bit later
            },
            "Forester_2": {
                # to be updated after calculating money_farmer and money_forester (!)
                "income": [self.model_param_dict["gdp_average"]],
                # to be updated after calculating money_farmer and money_forester (!)
                "bank_account": [
                    self.model_param_dict["bank_account_forestry_2"]
                ],
                # to be updated after calculating unemployment_rate after money_farmer and money_forester (!)
                "unemployment": [self.model_param_dict["unempl_rate_scotland"]],
                "area_sheep": [],  # to be updated a bit later
                "area_cattle": [0],  # to be updated a bit later
                "area_c_forest": [],  # to be updated a bit later
                "area_n_forest": [],  # to be updated a bit later
            },
            "SSDA": {
                # True/False for each game round, defaults to False for rounds 0, 1 and 2
                "ssda_block_choice": [None],
                # TODO: move to params
                "tourist_number": [4800],
                # TODO: move to params
                "tourist_factor": [1],
            },
        }

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

        # populate stakeholder-unspecific data stores
        unique_global, counts_global = np.unique(
            self.lulc_matrix_stack[:, :, 0], return_counts=True
        )
        d_unique_counts_global = dict(
            zip(unique_global[:-1], counts_global[:-1])
        )
        self.data_store["variable_model_params"]["area_sheep_total"].append(
            d_unique_counts_global[1]
        )
        self.data_store["variable_model_params"]["area_n_forest_total"].append(
            d_unique_counts_global[2]
        )
        self.data_store["variable_model_params"]["area_c_forest_total"].append(
            d_unique_counts_global[3]
        )

        # iteratively populate the stakeholder-specific data stores
        # TODO [low]: ultimatively remove hard-coding in the following sequence
        for (
            stakeholder_name,
            stakeholder_id,
        ) in dicts.stakeholder_id_dict.items():
            if stakeholder_name == "SSDA":
                continue

            # select data of property that stakeholder owns
            lulc_data_sel = self.lulc_matrix_stack[:, :, 0][
                self.property_rights_matrix == stakeholder_id
            ]

            # get unique value counts
            unique, counts = np.unique(lulc_data_sel, return_counts=True)
            d_unique_counts = dict(zip(unique[:-1], counts[:-1]))

            # init relevant data store parts
            self.data_store[stakeholder_name]["area_sheep"].append(
                d_unique_counts[1]
            )
            self.data_store[stakeholder_name]["area_n_forest"].append(
                d_unique_counts[2]
            )
            self.data_store[stakeholder_name]["area_c_forest"].append(
                d_unique_counts[3]
            )

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
            return (
                self.lulc_matrix_stack[:, :, self.current_round],
                self.remapping_dict,
            )
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

    def update_lulc_matrix_based_on_mgmt_decisions(
        self, current_round=None, seed=42
    ):

        # TODO [some time in the new year]:
        #  clean up and simplify, the code is really messy as of now

        # if map updating should always follow the same random process.
        # if seed=None, it changes for each function call.
        if seed is not None:
            np.random.seed(seed)

        # leaves us the freedom to explicitly choose a round,
        # but defaults to using the class parameter value
        if current_round is None:
            current_round = self.current_round

        # max pixels
        n_pixels_per_block_max = self.n_pixels_per_block ** 2
        n_pixels_max = self.n_pixels ** 2

        # query relevant mgmt decisions of current round
        mgmt_decisions_of_round = self.df_mgmt_decisions_long.query(
            "Round == {current_round}".format(current_round=current_round)
        )

        # all blocks
        blocks = self.block_definition_matrix_block_lvl.flatten()

        # copy lulc_matrix data from the matrix stack
        mar_field_2d = self.lulc_matrix_stack[
            :, :, self.current_round - 1
        ].copy()
        mar_field_1d = mar_field_2d.flatten()

        # iterate over blocks where lulc should be
        # updated based on mgmt decision
        # ---------------------------------------------------
        for block in blocks:

            # copy mar_field in each block iteration
            mar = mar_field_2d.copy()

            # query relevant mgmt decisions of current round
            # TODO: find tailored solution for SSDA
            mgmt_decisions_of_round_and_block = mgmt_decisions_of_round.query(
                "Plot == {block} & Player != 'SSDA'".format(block=block)
            )

            # continue to next block in case no mgmt decision has been made
            if mgmt_decisions_of_round_and_block.empty:
                continue
            else:

                # create a boolean mask which is True over the
                # pixels of the particular block
                mgmt_mask = self.block_definition_matrix_pixel_lvl == block

                # update the mask of the lulc_matrix based on the block mask
                mar[~mgmt_mask] = np.ma.masked

                # calculate number of counts per lulc category
                unique, counts = np.unique(mar, return_counts=True)

                # infer number of unmasked px of block (this is where the
                # we can actually update the lulc as I don't let them
                # expand any areas outside of the biosphere)
                n_pixels_unmasked, n_pixels_masked_total = (
                    counts[:-1],
                    counts[-1],
                )
                n_pixels_unmasked_in_block = (
                    n_pixels_max - n_pixels_masked_total
                )
                lulc_types_unmasked = unique[unique.mask == False].data

                # store info as dict
                block_lulc_distribution = dict(
                    zip(lulc_types_unmasked, n_pixels_unmasked)
                )

                # create index matrix
                mar_ix_1d = np.arange(mar.ravel().size)
                mar_ix_1d_masked = np.ma.masked_array(mar_ix_1d, mask=mar.mask)
                mar_ix_2d_masked = mar_ix_1d_masked.reshape(
                    self.n_pixels, self.n_pixels
                )

                # iterate through the lulc types
                for (
                    group_id,
                    group_data,
                ) in mgmt_decisions_of_round_and_block.groupby("Player"):

                    # ------------------------------------------------
                    # 1) Farmers Decisions
                    # ------------------------------------------------

                    # decision on cattle
                    cattle_farming_conversion_decision = (
                        group_data[group_data["lulc_category_id"] == 4]
                        .loc[:, "mgmt_decision"]
                        .values[0]
                    )

                    # decision on sheep: magnitude with opposite sign of cattle
                    sheep_farming_conversion_decision = (
                        cattle_farming_conversion_decision * -1
                    )

                    # farmer's decicions on native forest are evaluated only
                    # after the farming type conversions (if any)
                    native_forest_farmer_decision = (
                        group_data[group_data["lulc_category_id"] == 2]
                        .loc[:, "mgmt_decision"]
                        .values[0]
                    )

                    # A) check if action for farming type conversion is needed
                    # ------------------------------------------------
                    if np.isclose(
                        cattle_farming_conversion_decision, 0
                    ) or np.isnan(cattle_farming_conversion_decision):
                        # no action
                        pass
                    else:
                        # action

                        # update lulc matrix in block via random sampling
                        # of array elements without replacement

                        # case 1: convert from sheep to cattle
                        if cattle_farming_conversion_decision > 0:
                            convert_to = 4

                        # case 2: convert from cattle back to sheep
                        elif cattle_farming_conversion_decision < 0:
                            convert_to = 1
                        else:
                            # TODO: check
                            continue

                        # case 3: plant native forest, via random assignment of
                        # possibly all of sheep, cattle and native forest
                        # TODO

                        # random draw among those indices that are within
                        # the block and playing field

                        sample_size = int(
                            np.abs(cattle_farming_conversion_decision)
                        )

                        # respect an upper limit on the maximum sample size
                        if sample_size > n_pixels_unmasked_in_block:
                            sample_size = n_pixels_unmasked_in_block

                        ix_sel_for_draw = mar_ix_1d_masked[
                            mar_ix_1d_masked.mask == False
                        ]
                        random_ix_1d = np.random.choice(
                            a=ix_sel_for_draw, size=sample_size, replace=False
                        )

                        # update 1d copy of field lulc matrix within
                        # block at drawn indices
                        mar_field_1d[random_ix_1d] = convert_to

                    # B) check if action for conversion to more native
                    # forest is needed
                    # ------------------------------------------------

                    # TODO: comment out if crashing
                    if (
                        np.isclose(native_forest_farmer_decision, 0)
                        or np.isnan(native_forest_farmer_decision)
                        or native_forest_farmer_decision < 0
                    ):
                        # no action
                        pass
                    else:
                        # action

                        # update lulc matrix in block via random sampling
                        # of array elements without replacement
                        # ------------------------------------------------

                        # case 3: plant native forest, via random assignment of
                        # possibly all of sheep, cattle and native forest
                        convert_to = 2

                        # random draw among those indices that are within
                        # the block and playing field

                        sample_size = int(np.abs(native_forest_farmer_decision))

                        # respect an upper limit on the maximum sample size
                        if sample_size > n_pixels_unmasked_in_block:
                            sample_size = n_pixels_unmasked_in_block

                        ix_sel_for_draw = mar_ix_1d_masked[
                            mar_ix_1d_masked.mask == False
                        ]
                        random_ix_1d = np.random.choice(
                            a=ix_sel_for_draw, size=sample_size, replace=False
                        )

                        # update 1d copy of field lulc matrix within
                        # block at drawn indices
                        mar_field_1d[random_ix_1d] = convert_to

                    # ------------------------------------------------
                    # 2) Foresters Decisions
                    # ------------------------------------------------

                    # A) conversion from native forest to commercial forest
                    # ------------------------------------------------
                    native_forest_conversion_decision = (
                        group_data[group_data["lulc_category_id"] == 2]
                        .loc[:, "mgmt_decision"]
                        .values[0]
                    )
                    if np.isclose(
                        native_forest_conversion_decision, 0
                    ) or np.isnan(native_forest_conversion_decision):
                        pass
                    else:

                        # update lulc matrix in block via random sampling
                        # of array elements without replacement
                        # ------------------------------------------------

                        # determine which forest category to convert to
                        if native_forest_conversion_decision > 0:
                            convert_forest_to = 2
                        elif native_forest_conversion_decision < 0:
                            convert_forest_to = 3
                        else:
                            # TODO: check
                            continue

                        # random draw among those indices that are within
                        # the block and playing field
                        sample_size = int(
                            np.abs(native_forest_conversion_decision)
                        )

                        # adjust sample size based on boundary conditions
                        if sample_size > n_pixels_unmasked_in_block:
                            sample_size = n_pixels_unmasked_in_block

                        ix_sel_for_draw = mar_ix_1d_masked[
                            mar_ix_1d_masked.mask == False
                        ]
                        random_ix_1d = np.random.choice(
                            a=ix_sel_for_draw, size=sample_size, replace=False
                        )

                        # update 1d copy of field lulc matrix within
                        # block at drawn indices
                        mar_field_1d[random_ix_1d] = convert_forest_to

            # reshape 1d mar back to 2d
            mar_field_2d_updated = mar_field_1d.reshape(
                self.n_pixels, self.n_pixels
            )

            # update 3d
            mar_field_3d_updated = mar_field_2d_updated.reshape(
                mar_field_2d_updated.shape[0], mar_field_2d_updated.shape[1], 1
            )

            # obsolete:
            # plot the updated block after the block iteration is done
            # plt.imshow(mar, cmap=self.cmap_lulc)
            # plt.imshow(mar_field_2d_updated, cmap=self.cmap_lulc, alpha=1)
            # plt.imshow(mar_ix_2d_masked, alpha=0.5)

        # finally, add new lulc map to lulc_matrix_stack
        # --------------------------------------------------------
        try:
            self.lulc_matrix_stack = np.ma.append(
                a=self.lulc_matrix_stack, b=mar_field_3d_updated, axis=2
            )
        except UnboundLocalError as msg:
            print(
                "Error: {} \n"
                "-> Very likely, either the Plot Number or "
                "the Mgmt Decision is missing. Check all sheets!".format(msg)
            )

    def update_data_store(
        self, current_round=None, teamwork=None, ssda_choice=None
    ):
        # update income, unemployment, prices etc

        # leaves us the freedom to explicitly choose a round if needed
        if current_round is None:
            current_round = self.current_round

        # leaves us the freedom to explicitly choose if needed
        if teamwork is None:
            teamwork = model.teamwork(self.cooperation_matrix_block_lvl)

        # ---------------------------------------------------------------------
        # update stakeholder-unspecific area totals
        # ---------------------------------------------------------------------
        unique_global, counts_global = np.unique(
            self.lulc_matrix_stack[:, :, self.current_round], return_counts=True
        )

        d_unique_counts_global = dict(
            zip(unique_global[:-1], counts_global[:-1])
        )

        self.data_store["variable_model_params"]["area_sheep_total"].append(
            d_unique_counts_global[1]
        )
        self.data_store["variable_model_params"]["area_n_forest_total"].append(
            d_unique_counts_global[2]
        )
        self.data_store["variable_model_params"]["area_c_forest_total"].append(
            d_unique_counts_global[3]
        )
        self.data_store["variable_model_params"]["area_cattle_total"].append(
            d_unique_counts_global[4]
        )

        # ---------------------------------------------------------------------
        # update stakeholder-specific area totals
        # ---------------------------------------------------------------------

        # iteratively populate the stakeholder-specific data stores
        # TODO [low]: ultimatively remove hard-coding in the following sequence
        for (
            stakeholder_name,
            stakeholder_id,
        ) in dicts.stakeholder_id_dict.items():
            if stakeholder_name == "SSDA":
                continue

            # select data of property that stakeholder owns
            lulc_data_sel = self.lulc_matrix_stack[:, :, self.current_round][
                self.property_rights_matrix == stakeholder_id
            ]

            # get unique value counts
            unique, counts = np.unique(lulc_data_sel, return_counts=True)
            d_unique_counts = dict(zip(unique[:-1], counts[:-1]))

            # init relevant data store parts
            try:
                self.data_store[stakeholder_name]["area_sheep"].append(
                    d_unique_counts[1]
                )
            except KeyError as msg:
                print(msg)
                continue
            try:
                self.data_store[stakeholder_name]["area_n_forest"].append(
                    d_unique_counts[2]
                )
            except KeyError as msg:
                print(msg)
                continue
            try:
                self.data_store[stakeholder_name]["area_c_forest"].append(
                    d_unique_counts[3]
                )
            except KeyError as msg:
                print(msg)
                continue
            try:
                self.data_store[stakeholder_name]["area_cattle"].append(
                    d_unique_counts[4]
                )
            except KeyError as msg:
                print(msg)
                continue

        # ---------------------------------------------------------------------
        # teamwork update
        # ---------------------------------------------------------------------
        self.data_store["variable_model_params"]["teamwork"].append(teamwork)

        # ---------------------------------------------------------------------
        # Tourism update
        # ---------------------------------------------------------------------
        if ssda_choice is not None:
            tourism_mask = self.block_definition_matrix_pixel_lvl == ssda_choice

            # copy lulc_matrix data from stack
            mar = self.lulc_matrix_stack[:, :, self.current_round - 1].copy()

            # update the mask of the lulc_matrix based on the block mask
            mar[~tourism_mask] = np.ma.masked

            # calc
            number_tourists, tourism_factor = model.tourism_factor(
                tourism_factor_matrix=mar,
                gdp_tourism_factor=40,  # TODO: removing hard-coding needed?
            )

            # update lists
            self.data_store["SSDA"]["tourist_number"].append(number_tourists)
            self.data_store["SSDA"]["tourist_factor"].append(tourism_factor)

        # global update, independent of choice
        self.data_store["SSDA"]["ssda_block_choice"].append(ssda_choice)

        # ---------------------------------------------------------------------
        # Price update
        # ---------------------------------------------------------------------
        # yield in round 0
        _, tot_sheep, _, _ = model.yield_map(self.lulc_matrix_stack[:, :, 0])

        # yield in this round
        tot_cattle, _, tot_n_forest, tot_c_forest = model.yield_map(
            self.lulc_matrix_stack[:, :, self.current_round]
        )

        # calculate prices
        (
            cattle_price_new,
            sheep_price_new,
            n_forest_price_new,
            c_forest_price_new,
        ) = model.price_per_pixel(
            current_round=self.current_round,
            brexit=self.brexit_round,
            tot_sheep_0=self.data_store["variable_model_params"][
                "area_sheep_total"
            ][0],
            tot_cattle=self.data_store["variable_model_params"][
                "area_cattle_total"
            ][self.current_round],
            tot_n_forest=self.data_store["variable_model_params"][
                "area_n_forest_total"
            ][self.current_round],
            tot_c_forest=self.data_store["variable_model_params"][
                "area_c_forest_total"
            ][self.current_round],
            income_farmland_cattle=self.model_param_dict[
                "income_farmland_cattle"
            ],
            income_farmland_sheep=self.model_param_dict[
                "income_farmland_sheep"
            ],
            income_forest_native=self.model_param_dict["income_forest_native"],
            income_forest_commercial=self.model_param_dict[
                "income_forest_commercial"
            ],
        )

        # update prices
        self.data_store["variable_model_params"]["cattle_price"].append(
            cattle_price_new
        )
        self.data_store["variable_model_params"]["sheep_price"].append(
            sheep_price_new
        )
        self.data_store["variable_model_params"]["n_forest_price"].append(
            n_forest_price_new
        )
        self.data_store["variable_model_params"]["c_forest_price"].append(
            c_forest_price_new
        )

        # ---------------------------------------------------------------------
        # Income update
        # ---------------------------------------------------------------------

        # Farmers
        # ---------------------------------------------------------------------
        for farmer_player in ["Farmer_1", "Farmer_2"]:
            farmer_earning, farmer_bank_current = model.money_farmer(
                current_round=self.current_round,
                tourism_factor=self.data_store["SSDA"]["tourist_factor"][
                    self.current_round
                ],
                teams=self.teamwork_round,  # fixed
                brexit=self.brexit_round,
                teamwork=self.data_store["variable_model_params"]["teamwork"][
                    self.current_round
                ],
                area_sheep=self.data_store[farmer_player]["area_sheep"],
                area_cattle=self.data_store[farmer_player]["area_cattle"],
                area_c_forest=self.data_store[farmer_player]["area_c_forest"],
                area_n_forest=self.data_store[farmer_player]["area_n_forest"],
                sheep_price=self.data_store["variable_model_params"][
                    "sheep_price"
                ],
                cattle_price=self.data_store["variable_model_params"][
                    "cattle_price"
                ],
                n_forest_price=self.data_store["variable_model_params"][
                    "n_forest_price"
                ],
                c_forest_price=self.data_store["variable_model_params"][
                    "c_forest_price"
                ],
                bank_account_farmer_1=self.data_store[farmer_player][
                    "bank_account"
                ][0],
                # bank account shared in round 0
                bank=self.data_store[farmer_player]["bank_account"][
                    self.current_round - 1
                ],
                # state of bank account in last round
                gdp_pc_scotland=self.model_param_dict["gdp_average"],
            )

            # update data store
            self.data_store[farmer_player]["income"].append(farmer_earning)
            self.data_store[farmer_player]["bank_account"].append(
                farmer_bank_current
            )

        # Foresters
        # ---------------------------------------------------------------------
        for forester_player in ["Forester_1", "Forester_2"]:
            forester_earning, forester_bank_current = model.money_forester(
                current_round=self.current_round,
                tourism_factor=self.data_store["SSDA"]["tourist_factor"][
                    self.current_round
                ],
                teams=self.teamwork_round,  # fixed
                brexit=self.brexit_round,
                teamwork=self.data_store["variable_model_params"]["teamwork"][
                    self.current_round
                ],
                area_sheep=self.data_store[forester_player]["area_sheep"],
                area_c_forest=self.data_store[forester_player]["area_c_forest"],
                area_n_forest=self.data_store[forester_player]["area_n_forest"],
                sheep_price=self.data_store["variable_model_params"][
                    "sheep_price"
                ],
                n_forest_price=self.data_store["variable_model_params"][
                    "n_forest_price"
                ],
                c_forest_price=self.data_store["variable_model_params"][
                    "c_forest_price"
                ],
                bank_account_forestry_1=self.data_store[forester_player][
                    "bank_account"
                ][0],
                # bank account shared in round 0
                bank=self.data_store[forester_player]["bank_account"][
                    self.current_round - 1
                ],
                # state of bank account in last round
                gdp_pc_scotland=self.model_param_dict["gdp_average"],
            )

            # update data store
            self.data_store[forester_player]["income"].append(forester_earning)
            self.data_store[forester_player]["bank_account"].append(
                forester_bank_current
            )

        # ---------------------------------------------------------------------
        # Unemployment update
        # ---------------------------------------------------------------------
        # IMPORTANT: Martina's code: **Earning = Income** !!!
        #
        # - new earning should be added to **income** lists
        # - new bank account status should be added to **bank_account** lists
        # ---------------------------------------------------------------------
        for player in ["Farmer_1", "Farmer_2", "Forester_1", "Forester_2"]:
            new_unemployment = model.unemployment_rate(
                earning=self.data_store[player]["income"][
                    self.current_round - 1
                ],
                earning_new=self.data_store[player]["income"][
                    self.current_round
                ],
                unemployment=self.data_store[player]["unemployment"][
                    self.current_round - 1
                ],
            )
            # update data store
            self.data_store[player]["unemployment"].append(new_unemployment)

    def show(self, granularity=1, figure_size=None, ax=None, cattle=False):
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
            cmap=self.cmap_lulc if cattle is True else self.cmap_lulc_no_cattle,
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

        # display quadrant property
        for name, pos in dicts.stakeholder_name_pos_dict.items():
            ax.text(
                x=pos[0],
                y=pos[1],
                s=name,
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                color="black",
                fontsize=14,
                bbox=dict(facecolor="white", alpha=0.7),
            )

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

        # strip class names
        classes = base.convert_lulc_id_to_class(
            int_array=unique[:-1], mapping=self.simplified_lulc_mapping
        )
        classes_short = [class_name.split(" ")[0] for class_name in classes]

        # plot
        ax.bar(x=classes_short, height=bar_width, color=self.cmap_lulc_hex)
        ax.grid(which="major", axis="y", linestyle="--")

        return ax

    def show_dashboard(
        self,
        granularity=1,
        figure_size=None,
        property_rights=False,
        relative=False,
        cattle=False,
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
        self.show(granularity=granularity, ax=ax_map, cattle=cattle)

        # PROPERTY
        self.show_bar(granularity=granularity, relative=relative, ax=ax_bar)

        # GDP & UNEMPLOYMENT of current round (!)#
        # ---------------------------------------------------------------------
        vis_data_dict = {}

        for var in ["bank_account", "income", "unemployment"]:
            d = {}
            for player in self.players[:-1]:
                d[player] = self.data_store[player][var][self.current_round]

            # insert
            vis_data_dict[var] = d

        vis_data = pd.concat(
            {
                k: pd.DataFrame(v, index=["value"]).T
                for k, v in vis_data_dict.items()
            },
            axis=0,
        )

        # PLOT BANK ACCOUNTS
        # ---------------------------------------------------------------------
        df_bank_account = (
            vis_data.iloc[vis_data.index.get_level_values(0) == "bank_account"]
            .reset_index(level=[0, 1])
            .drop("level_0", axis=1)
        )
        df_bank_account.plot.bar(
            ax=ax_gdp, x="level_1", y="value", legend=False, alpha=0.8
        )

        ax_gdp.axhline(0, linestyle="--", linewidth=1)
        ax_gdp.set_ylabel("Bank Account ($)")
        ax_gdp.set_xlabel(None)
        ax_gdp.grid(which="major", axis="y", linestyle="--")
        ax_gdp.set_xticklabels([])

        # PLOT UNEMPLOYMENT
        # ---------------------------------------------------------------------
        df_unempl = (
            vis_data.iloc[vis_data.index.get_level_values(0) == "unemployment"]
            .reset_index(level=[0, 1])
            .drop("level_0", axis=1)
        )

        # linear scaling
        df_unempl["value_reverse"] = 100 - df_unempl["value"]

        # plot
        df_unempl.plot.bar(
            ax=ax_empl,
            x="level_1",
            y="value_reverse",
            legend=False,
            color="#e6550d",
            alpha=0.7,
        )
        ax_empl.axhline(0, linestyle="--", linewidth=1)
        ax_empl.set_ylabel("Employees (MAX = 50)")
        ax_empl.set_yscale("log")
        ax_empl.set_xlabel(None)
        # ax_empl.set_ylim(1, 100)

        # remove all y tick labels
        ax_empl.set_yticks([], minor=False)  # major
        ax_empl.set_yticks([], minor=True)  # minor

        # labeling
        # ---------------------------------------------------------------------
        ax_bar.set_ylabel("Distribution of Parcels")

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
    from pathlib import Path

    # define dirs
    abspath = os.path.abspath("")
    project_dir = str(Path(abspath).parents[1])
    data_raw = os.path.join(project_dir, "data", "raw")
    data_processed = os.path.join(project_dir, "data", "processed")
    figure_dir = os.path.join(project_dir, "plots")

    # playing field size
    rows = cols = 80

    # set path to raster data
    fpath_map = os.path.join(
        data_processed, "NFI_rasterized_{}_{}.tif".format(rows, cols)
    )

    # load map
    field = MatrixGame(
        fpath=fpath_map,
        original_lulc_mapping=original_lulc_mapping,
        simplified_lulc_mapping=simplified_lulc_mapping,
        lulc_remapping=lulc_remapping,
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
