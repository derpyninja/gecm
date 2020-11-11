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
# TODO [low]: copy this dict into a new excel file to be parsed when needed
#  (also include long category descriptors!)
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
    "Other vegetation": 211
}

# TODO: shrub (7) in "Livestock Farming" OK?
nfi_mapping_v2 = {
    "Livestock Farming": [7, 13, 22, 23, 211],
    "Native Forest": [2, 4, 6, 9],
    "Commercial Forest": [1, 3, 8, 14, 15, 16],
    "Water": [21, 26],
    "Urban": [24, 25],
    "Other": [28, 29, 210]
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
