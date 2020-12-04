import seaborn as sns
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


def show_mgmt_decisions(
    df,
    current_round=None,
    cols=None,
    kind="bar",
    stacked=True,
    box_width_scaling=0.7,
    figure_size=(10, 8),
):
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
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # labels
    ax.set_ylabel("Management decisions (%)")
    ax.axhline(linestyle="-", color="grey")
    ax.set_title("Round {}".format(current_round))

    # aggregate decisions per stakeholder group are bounded by [-100, +100] %
    ax.set_ylim(-100, 100)
    return ax


def show_all_mgmt_decisions(df_mgmt_decisions_long):
    # https://seaborn.pydata.org/tutorial/relational.html: seaborn supports semantics of hue, size, and style
    g = sns.catplot(
        x="round",
        y="value",
        hue="variable",
        col="stakeholder",
        kind="bar",
        data=df_mgmt_decisions_long,
        saturation=0.5,
        ci=None,
        aspect=0.6,
    )

    (
        g.set_axis_labels("Round", "Management Decision (%)")
        # .set_xticklabels(["round", "round", "round"])
        # .set_titles("{col_name} {col_var}")
        .set(ylim=(-55, 55)).despine(left=False)
    )

    return g


def create_dummy_gridspec():
    """
    Testing purposes.

    Returns
    -------

    """
    from matplotlib.gridspec import GridSpec

    def format_axes(fig):
        for i, ax in enumerate(fig.axes):
            ax.text(0.5, 0.5, "ax%d" % (i + 1), va="center", ha="center")
            ax.tick_params(labelbottom=False, labelleft=False)

    fig = plt.figure(constrained_layout=True)

    gs = GridSpec(4, 3, figure=fig)
    ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=2, rowspan=4))
    ax2 = plt.subplot(gs.new_subplotspec((0, 2), colspan=1, rowspan=2))
    ax3 = plt.subplot(gs.new_subplotspec((2, 2), colspan=1, rowspan=2))

    fig.suptitle("GridSpec")
    format_axes(fig)

    plt.show()


if __name__ == "__main__":
    create_dummy_gridspec()
    plt.show()
