import numpy as np
import matplotlib as mpl
from copy import deepcopy
from src.gecm.dicts import original_lulc_mapping


def invert_dict(d):
    """
    Invert dictionary by switching keys and values.

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


def remap_array_with_dict(input_array, mapping):
    """
    TODO

    Parameters
    ----------
    input_array
    mapping

    Returns
    -------
    TODO
    """
    k = np.array(list(mapping.keys()))
    v = np.array(list(mapping.values()))

    out = np.ma.masked_all_like(input_array)
    print(out)

    for key, val in zip(k, v):
        out[input_array == key] = val

    return out


def convert_lulc_id_to_class(
    int_array, mapping=original_lulc_mapping, max_string_length=40
):
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
    """
    TODO

    Parameters
    ----------
    old_dict
    remap_dict
    remap_dict_ids
    res_dict

    Returns
    -------
    TODO
    """

    # define placeholder dict
    res_dict = res_dict or {}

    # invert id dict
    remap_dict_ids_inv = invert_dict(remap_dict_ids)

    # iterate over items of original mapping
    for old_lulc_name, old_lulc_code in old_dict.items():

        # iterate over items of remapping dict. Each row takes the following
        # form: {'newkey1': [oldvalue1, oldvalue2, ...], ...}
        for new_lulc_code, old_lulc_codes in remap_dict.items():
            new_lulc_name = remap_dict_ids_inv[new_lulc_code]

            if old_lulc_code in old_lulc_codes:
                res_dict[old_lulc_name] = new_lulc_code

    return res_dict


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
