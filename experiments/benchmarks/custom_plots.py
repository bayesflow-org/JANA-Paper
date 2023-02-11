import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from bayesflow.computational_utilities import simultaneous_ecdf_bands
import seaborn as sns

def plot_sbc_ecdf(
    post_samples,
    prior_samples,
    difference=False,
    stacked=False,
    fig_size=None,
    param_names=None,
    label_fontsize=24,
    legend_fontsize=24,
    title_fontsize=16,
    rank_ecdf_colors=["#009900", "#990000"],
    fill_color="grey",
    legend_spacing=0.8,
    ylim = None,
    **kwargs,
):
    # Store reference to number of parameters
    n_params = post_samples.shape[-1]

    # Compute fractional ranks (using broadcasting)
    ranks = np.sum(post_samples < prior_samples[:, np.newaxis, :], axis=1) / post_samples.shape[1]

    # Prepare figure
    f, ax = plt.subplots(1, 1, figsize=fig_size)

    patches = [None] * ranks.shape[-1]
    # Plot individual ecdf of parameters
    for j in range(ranks.shape[-1]):

        ecdf_single = np.sort(ranks[:, j])
        xx = ecdf_single
        yy = np.arange(1, xx.shape[-1] + 1) / float(xx.shape[-1])

        # Difference, if specified
        if difference:
            yy -= xx

        ax.plot(xx, yy, color=rank_ecdf_colors[j], alpha=0.95, **kwargs.pop("ecdf_line_kwargs", {}))
        patches[j] = mpatches.Rectangle([0, 0], 0.1, 0.1, facecolor=rank_ecdf_colors[j], label=param_names[j])
        
    # Compute uniform ECDF and bands
    alpha, z, L, H = simultaneous_ecdf_bands(post_samples.shape[0], **kwargs.pop("ecdf_bands_kwargs", {}))
    if ylim is not None:
        ax.set_ylim(ylim)

    # Difference, if specified
    if difference:
        L -= z
        H -= z

    # Add simultaneous bounds
    titles = [None]
    axes = [ax]

    ax.set_box_aspect(1)
    for _ax, title in zip(axes, titles):
        _ax.fill_between(z, L, H, color=fill_color, alpha=0.2)

        # Prettify plot
        sns.despine(ax=_ax)
        _ax.grid(alpha=0.35)
        _ax.legend(fontsize=legend_fontsize, loc='lower center', ncol=2, handles=patches, 
                   columnspacing=legend_spacing, handletextpad=0.3, handlelength=1, handleheight=1)
        _ax.set_xlabel("Fractional rank statistic", fontsize=label_fontsize)
        if difference:
            ylab = "ECDF difference"
        else:
            ylab = "ECDF"
        _ax.set_ylabel(ylab, fontsize=label_fontsize)
        _ax.set_title(title, fontsize=title_fontsize)

    f.tight_layout()
    return f