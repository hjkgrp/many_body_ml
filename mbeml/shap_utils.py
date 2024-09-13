import shap
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from shap.plots import colors
from shap.plots._utils import convert_ordering, convert_color


def combine_features(
    shap_values: shap.Explanation,
    features_to_combine: List[str],
    new_feature_name: str,
    combine_values=lambda x: np.sum(x, axis=1),
    combine_data=lambda x: np.sum(x, axis=1),
):
    # Combine
    mask = np.array([f in features_to_combine for f in shap_values.feature_names])
    # Check that all features in features_to_combine are actually part of the original
    # feature list
    if np.count_nonzero(mask) != len(features_to_combine):
        raise ValueError(
            f"Not all features_to_combine present in feature_names: {features_to_combine}"
        )
    # First remove the two features
    old_shape = shap_values.values.shape
    summed_values = np.zeros(
        (old_shape[0], old_shape[1] - len(features_to_combine) + 1)
    )
    summed_values[:, :-1] = shap_values.values[:, ~mask]
    summed_data = np.zeros_like(summed_values)
    summed_data[:, :-1] = shap_values.data[:, ~mask]
    summed_feature_names = [f for f, b in zip(shap_values.feature_names, mask) if not b]
    # Then append the combined feature
    summed_values[:, -1] = combine_values(shap_values.values[:, mask])
    summed_data[:, -1] = combine_data(shap_values.data[:, mask])
    summed_feature_names.append(new_feature_name)

    return shap.Explanation(
        values=summed_values,
        base_values=shap_values.base_values,
        data=summed_data,
        feature_names=summed_feature_names,
    )


def combine_ohe_to_int(x):
    """Combines One-Hot-Encoding back to integer encoding"""
    n_features = x.shape[1]
    int_features = np.arange(n_features)
    return np.sum(x * int_features, axis=1)


def sum_core_features(shap_values: shap.Explanation):
    result = combine_features(
        shap_values, ["M(II)", "M(III)"], "OS", combine_data=combine_ohe_to_int
    )
    result = combine_features(
        result,
        ["d3", "d4", "d5", "d6", "d7"],
        r"d$^\mathrm{n}$",
        combine_data=combine_ohe_to_int,
    )
    return result


def sum_ligand_features(shap_values: shap.Explanation):
    result = sum_core_features(shap_values)

    result = combine_features(
        result,
        [f"lig{i}_charge" for i in range(1, 7)],
        "$_\\mathrm{{lig}}$q",
    )

    for prime in ["", "^\\prime"]:
        props = (
            ["$\\chi$", "Z", "I", "T", "S"]
            if prime == ""
            else ["$\\chi$", "Z", "T", "S"]
        )
        for prop in props:
            depths = range(4) if prime == "" and prop != "I" else range(1, 4)
            for depth in depths:
                result = combine_features(
                    result,
                    [f"lig{i}_{prop}${prime}_{depth}$" for i in range(1, 7)],
                    f"{prop}${prime}_{depth}$",
                )
    return result


def custom_beeswarm(
    shap_values,
    ax=None,
    max_display=10,
    order=shap.Explanation.abs.mean(0),
    color=None,
    alpha=1,
    log_scale=False,
    s=16,
    group_remaining_features: bool = True,
):
    """adapted from https://github.com/shap/shap/blob/v0.46.0/shap/plots/_beeswarm.py"""
    if ax is None:
        ax = plt.gca()

    shap_exp = shap_values
    # we make a copy here, because later there are places that might modify this array
    values = np.copy(shap_exp.values)
    features = shap_exp.data
    feature_names = shap_exp.feature_names

    order = convert_ordering(order, values)
    color = convert_color(colors.red_blue)

    num_features = values.shape[1]

    # determine how many top features we will plot
    if max_display is None:
        max_display = len(feature_names)
    num_features = min(max_display, len(feature_names))

    # iteratively merge nodes until we can cut off the smallest feature values to stay within
    # num_features without breaking a cluster tree
    orig_inds = [[i] for i in range(len(feature_names))]
    orig_values = values.copy()

    feature_order = convert_ordering(order, shap.Explanation(np.abs(values)))
    # here we build our feature names, accounting for the fact that some features might be merged together
    feature_inds = feature_order[:max_display]
    feature_names_new = []
    for pos, inds in enumerate(orig_inds):
        if len(inds) == 1:
            feature_names_new.append(feature_names[inds[0]])
        elif len(inds) <= 2:
            feature_names_new.append(" + ".join([feature_names[i] for i in inds]))
        else:
            max_ind = np.argmax(np.abs(orig_values).mean(0)[inds])
            feature_names_new.append(
                feature_names[inds[max_ind]] + " + %d other features" % (len(inds) - 1)
            )
    feature_names = feature_names_new

    # see how many individual (vs. grouped at the end) features we are plotting
    include_grouped_remaining = (
        num_features < len(values[0]) and group_remaining_features
    )
    if include_grouped_remaining:
        num_cut = np.sum(
            [
                len(orig_inds[feature_order[i]])
                for i in range(num_features - 1, len(values[0]))
            ]
        )
        values[:, feature_order[num_features - 1]] = np.sum(
            [
                values[:, feature_order[i]]
                for i in range(num_features - 1, len(values[0]))
            ],
            0,
        )

    # build our y-tick labels
    yticklabels = [feature_names[i] for i in feature_inds]
    if include_grouped_remaining:
        yticklabels[-1] = "Sum of %d other features" % num_cut

    row_height = 0.4
    ax.axvline(x=0, color="k", zorder=-1)

    # make the beeswarm dots
    for pos, i in enumerate(reversed(feature_inds)):
        ax.axhline(y=pos, color="#cccccc", lw=0.75, dashes=(1, 5), zorder=-1)
        shaps = values[:, i]
        fvalues = features[:, i]
        inds = np.arange(len(shaps))

        np.random.shuffle(inds)
        if fvalues is not None:
            fvalues = fvalues[inds]
        shaps = shaps[inds]

        N = len(shaps)
        nbins = 100
        quant = np.round(
            nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8)
        )
        inds = np.argsort(quant + np.random.randn(N) * 1e-6)
        layer = 0
        last_bin = -1
        ys = np.zeros(N)
        for ind in inds:
            if quant[ind] != last_bin:
                layer = 0
            ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]
        ys *= 0.9 * (row_height / np.max(ys + 1))

        # trim the color range, but prevent the color range from collapsing
        vmin = np.nanpercentile(fvalues, 5)
        vmax = np.nanpercentile(fvalues, 95)
        if vmin == vmax:
            vmin = np.nanpercentile(fvalues, 1)
            vmax = np.nanpercentile(fvalues, 99)
            if vmin == vmax:
                vmin = np.min(fvalues)
                vmax = np.max(fvalues)
        if vmin > vmax:  # fixes rare numerical precision issues
            vmin = vmax

        # plot the non-nan fvalues colored by the trimmed feature value
        cvals = fvalues.astype(np.float64)
        cvals_imp = cvals.copy()
        cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
        cvals[cvals_imp > vmax] = vmax
        cvals[cvals_imp < vmin] = vmin
        ax.scatter(
            shaps,
            pos + ys,
            cmap=color,
            vmin=vmin,
            vmax=vmax,
            s=s,
            c=cvals,
            alpha=alpha,
            linewidth=0,
            zorder=3,
            rasterized=len(shaps) > 500,
        )

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_yticks(range(len(feature_inds)), reversed(yticklabels), va="center")
    ax.tick_params("y", length=20, width=0.5, which="major")
    ax.tick_params("x")
    ax.set_ylim(-1, len(feature_inds))


def draw_colorbar(
    ax,
):
    # draw the color bar
    color = colors.red_blue
    import matplotlib.cm as cm

    m = cm.ScalarMappable(cmap=color)
    m.set_array([0, 1])
    cb = plt.colorbar(m, ax=ax, ticks=[0, 1], aspect=40, pad=0.02)
    cb.set_ticklabels(["Low", "High"])
    cb.set_label("Feature value", labelpad=-8)
    cb.ax.tick_params(length=0)
    cb.set_alpha(1)
    cb.outline.set_visible(False)
