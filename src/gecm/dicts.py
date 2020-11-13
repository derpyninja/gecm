import numpy as np
from copy import deepcopy

# ---------------------------------------------
# Category - ID Mapping
# ---------------------------------------------
# Source: NFI woodland map 2018
# 2 Categories: Woodland & Non Woodland
# Woodland categories run from 1-16
# Non Woodland categories run from 1-11,
# but the number 2 precludes them to
# differentiate them from woodland categories
# ---------------------------------------------
# TODO 1 [low]: copy this dict into a new excel file to be parsed when needed
#  (also include long category descriptors!)
# TODO 2 [low]: add np.nan to mapping as "outside of biosphere" or so
nfi_mapping = {
    "Conifer": 1,                   # start of woodland categories
    "Broadleaved": 2,
    "Mixed mainly conifer": 3,
    "Mixed mainly broadleaved": 4,
    "Coppice with standards": 6,
    "Shrub": 7,
    "Young trees": 8,
    "Felled": 9,
    "Ground prep": 10,
    "Low density": 13,
    "Assumed woodland": 14,
    "Failed": 15,
    "Windblow": 16,
    "Open water": 21,               # start of non-woodland categories
    "Grassland": 22,
    "Agriculture land": 23,
    "Urban": 24,
    "Road": 25,
    "River": 26,
    "Quarry": 28,
    "Bare area": 29,
    "Windfarm": 210,
    "Other vegetation": 211,
    "Grazing": 240    # defined by us
}

# Q&D
nfi_mapping_simplified = {'Conifer': 3, 'Broadleaved': 2, 'Mixed mainly conifer': 3, 'Mixed mainly broadleaved': 2, 'Coppice with standards': 2, 'Shrub': 1, 'Young trees': 3, 'Felled': 2, 'Low density': 1, 'Assumed woodland': 3, 'Failed': 3, 'Windblow': 3, 'Open water': 4, 'Grassland': 1, 'Agriculture land': 1, 'Urban': 5, 'Road': 5, 'River': 4, 'Quarry': 6, 'Bare area': 6, 'Windfarm': 6, 'Other vegetation': 1, 'Grazing': 1}

# TODO 3: shrub (7) in "Livestock Farming" OK?
nfi_mapping_v2 = {
    1: [7, 13, 22, 23, 211, 240],
    2: [2, 4, 6, 9],
    3: [1, 3, 8, 14, 15, 16],
    4: [21, 26],
    5: [24, 25],
    6: [28, 29, 210]
}

id_mapping = {
    "Livestock Farming": 1,
    "Native Forest": 2,
    "Commercial Forest": 3,
    "Water": 4,
    "Urban": 5,
    "Other": 6
}


def invert_dict(d):
    """
    Invert dictionary.

    Parameters
    ----------
    d : dict
        python dictionary

    Returns
    -------
    dict
        Inverted python dictionary
    """
    return dict((v, k) for k, v in d.items())


def int2class(int_array, mapping=nfi_mapping, max_string_length=40):
    """
    Map integers to LULC classes.

    Parameters
    ----------
    mapping : dict
        Describes mapping between ints and strings of LULC classes.
    int_array : np.array (int)
        2D Array representing a LULC map via integers
    max_string_length : int
        Int to str conversion needs a max str length param

    Returns
    -------
    np.array (str)
        2D Array representing a LULC map via classes
    """
    class_array = deepcopy(int_array).astype("S{}".format(max_string_length))
    boolean_mask = int_array.mask
    inverse_nfi_mapping = invert_dict(mapping)

    # does work for np.ma
    if int_array.ndim > 1:
        # re-map
        for (index, val) in np.ndenumerate(int_array):
            masked = boolean_mask[index]
            if not masked:
                class_array[index[0], index[1]] = inverse_nfi_mapping[val]

        return class_array

    else:
        # currently does not work for np.ma
        return np.array([inverse_nfi_mapping[i] for i in int_array])


def remap_lulc_dict(old_dict, remap_dict, remap_dict_ids, res_dict=None):

    # define placeholder dict
    res_dict = res_dict or {}

    # invert id dict
    remap_dict_ids_inv = invert_dict(remap_dict_ids)

    # iterate over items of original mapping
    for old_lulc_name, old_lulc_code in old_dict.items():
        print(old_lulc_name, old_lulc_code)

        # iterate over items of remapping dict. Each row takes the following
        # form: {'newkey1': [oldvalue1, oldvalue2, ...], ...}
        for new_lulc_code, old_lulc_codes in remap_dict.items():
            new_lulc_name = remap_dict_ids_inv[new_lulc_code]
            print(new_lulc_name, old_lulc_codes)

            if old_lulc_code in old_lulc_codes:
                res_dict[old_lulc_name] = new_lulc_code

    return res_dict


if __name__ == "__main__":

    nfi_remapped = remap_lulc_dict(
        old_dict=nfi_mapping,
        remap_dict=nfi_mapping_v2,
        remap_dict_ids=id_mapping
    )

    print(nfi_remapped)




