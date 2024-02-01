import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.markers import MarkerStyle

color_dict = {"cr": "C0", "mn": "C1", "fe": "C2", "co": "C3"}
marker_dict = {"2": "o", "3": "d"}


def core_legend(ax, scatter_kwargs=None, legend_kwargs=None):
    if scatter_kwargs is None:
        scatter_kwargs = dict(alpha=0.5, edgecolors="none", s=20)
    if legend_kwargs is None:
        legend_kwargs = dict(loc="upper left")

    handles = []
    labels = []
    for metal, color in color_dict.items():
        handles.append(
            (
                ax.scatter(
                    [], [], color=color, marker=marker_dict["2"], **scatter_kwargs
                ),
                ax.scatter(
                    [], [], color=color, marker=marker_dict["3"], **scatter_kwargs
                ),
            )
        )
        labels.append(metal.capitalize())
    for ox, marker in [("II", "o"), ("III", "d")]:
        handles.append(
            ax.scatter(
                [],
                [],
                color="none",
                marker=marker,
                linewidth=0.5,
                **dict(scatter_kwargs, edgecolors="k", alpha=1.0),
            )
        )
        labels.append(ox)

    legend = ax.legend(
        handles,
        labels,
        ncols=6,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=-0.1)},
        **legend_kwargs,
    )
    return legend


def scatter_random_z(
    ax, x, y, colors=None, markers=None, rng=np.random.default_rng(0), **scatter_kwargs
):
    """Solves the issue of layered scatter plots by choosing a random order and plotting
    all points individually

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot into.
    x : numpy.ndarray
        Array of x values
    y : numpy.ndarray
        Array of y values
    colors : numpy.ndarray, optional
        Array of colors, by default None
    markers : numpy.ndarray, optional
        Array of markers, by default None
    """
    indices = rng.permutation(range(len(x)))
    for i in indices:
        ax.scatter(x[i], y[i], color=colors[i], marker=markers[i], **scatter_kwargs)


def interpolation_plot(ax, x, y, color, alpha_cis=0.4, alpha_trans=0.8, **plot_kwargs):
    # Solid lines from homoleptic to 5+1
    ax.plot(x[:2], y[:2], color=color, **plot_kwargs)
    ax.plot(x[8:11], y[8:11], color=color, **plot_kwargs)
    ax.plot(x[17:20], y[17:20], color=color, **plot_kwargs)
    ax.plot(x[26:], y[26:], color=color, **plot_kwargs)

    for inds in [[1, 2, 4, 6, 8], [10, 11, 13, 15, 17], [19, 20, 22, 24, 26]]:
        # Lines through cis and mer isomers
        ax.plot(
            x[inds],
            y[inds],
            color=color,
            linestyle=(0, (2, 1)),
            **dict(plot_kwargs, alpha=alpha_cis),
        )
        # Second line through trans and fac isomers
        # Shift inner 3 indices by one
        inds = [inds[0], inds[1] + 1, inds[2] + 1, inds[3] + 1, inds[4]]
        ax.plot(
            x[inds],
            y[inds],
            color=color,
            linestyle=(1, (1, 2)),
            **dict(plot_kwargs, alpha=alpha_trans),
        )


def interpolation_scatter(
    ax, x, y, color, marker, alpha_cis=0.4, alpha_trans=0.8, **scatter_kwargs
):
    x_duplicates = np.zeros_like(x, dtype=bool)
    x_duplicates[2:8] = True
    x_duplicates[11:17] = True
    x_duplicates[20:26] = True
    ax.scatter(
        x[~x_duplicates],
        y[~x_duplicates],
        color=color,
        marker=marker,
        **scatter_kwargs,
    )
    # cis/fac points
    ax.scatter(
        x[x_duplicates][::2],
        y[x_duplicates][::2],
        color=color,
        marker=MarkerStyle(marker, fillstyle="left"),
        **dict(scatter_kwargs, alpha=alpha_cis),
    )
    # trans/mer points
    ax.scatter(
        x[x_duplicates][1::2],
        y[x_duplicates][1::2],
        color=color,
        marker=MarkerStyle(marker, fillstyle="right"),
        **dict(scatter_kwargs, alpha=alpha_trans),
    )
