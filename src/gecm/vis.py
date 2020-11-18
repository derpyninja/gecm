import matplotlib as mpl
import matplotlib.pyplot as plt


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


def show_mgmt_decisions(df, current_round=None, cols=None, kind='bar', stacked=True, box_width_scaling=0.7, figure_size=(10, 8)):
    # based on https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot

    # allow subsets
    if cols is not None:
        df = df[cols]

    # plot decisions
    fig, ax = plt.subplots(figsize=figure_size)
    df.plot(ax=ax, kind=kind, stacked=stacked)

    # Shrink current axis & put a legend to the right of the current axis
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * box_width_scaling, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # labels
    ax.set_ylabel("Management decisions (%)")
    ax.axhline(linestyle="-", color="grey")
    ax.set_title("Round {}".format(current_round))

    # aggregate decisions per stakeholder group are bounded by [-100, +100] %
    ax.set_ylim(-100, 100)

    return ax
