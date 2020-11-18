import matplotlib as mpl


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
