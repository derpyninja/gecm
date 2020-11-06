import numpy as np
from copy import deepcopy


# implements the detailed NFI mapping
nfi_mapping = {
    "Bare area": 1,
    "Agriculture land": 2,
    "Urban": 3,
    "Grassland": 4,
    "Quarry": 5,
    "Road": 6,
    "Other vegetation": 7,
    "River": 8,
    "Open water": 9,
    "Windfarm": 10,
    "Assumed woodland": 11,
    "Broadleaved": 12,
    "Conifer": 13,
    "Felled": 14,
    "Failed": 15,
    "Ground prep": 16,
    "Low density": 17,
    "Mixed mainly broadleaved": 18,
    "Mixed mainly conifer": 19,
    "Young trees": 20,
    "Coppice with standards": 21,
    "Shrub": 22,
    "Windblow": 23,
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
