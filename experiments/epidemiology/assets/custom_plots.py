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

def plot_sbc_ecdf_appendix(
    post_samples,
    prior_samples,
    difference=False,
    stacked=False,
    fig_size=None,
    param_names=None,
    label_fontsize=30,
    legend_fontsize=30,
    title_fontsize=30,
    rank_ecdf_color="#a34f4f",
    fill_color="grey",
    remove_plots = None,
    **kwargs,
):
    """Creates the empirical CDFs for each marginal rank distribution and plots it against
    a uniform ECDF. ECDF simultaneous bands are drawn using simulations from the uniform. Inspired by:
    [1] Säilynoja, T., Bürkner, P. C., & Vehtari, A. (2022). Graphical test for discrete uniformity and
    its applications in goodness-of-fit evaluation and multiple sample comparison. Statistics and Computing,
    32(2), 1-21. https://arxiv.org/abs/2103.10522
    For models with many parameters, use `stacked=True` to obtain an idea of the overall calibration
    of a posterior approximator.
    Parameters
    ----------
    post_samples      : np.ndarray of shape (n_data_sets, n_post_draws, n_params)
        The posterior draws obtained from n_data_sets
    prior_samples     : np.ndarray of shape (n_data_sets, n_params)
        The prior draws obtained for generating n_data_sets
    difference        : bool, optional, default: False
        If `True`, plots the ECDF difference. Enables a more dynamic visualization range.
    stacked           : bool, optional, default: False
        If `True`, all ECDFs will be plotted on the same plot. If `False`, each ECDF will
        have its own subplot, similar to the behavior of `plot_sbc_histograms`.
    param_names       : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None. Only relevant if `stacked=False`.
    fig_size          : tuple or None, optional, default: None
        The figure size passed to the matplotlib constructor. Inferred if None.
    label_fontsize    : int, optional, default: 14
        The font size of the y-label and y-label texts
    legend_fontsize   : int, optional, default: 14
        The font size of the legend text
    title_fontsize    : int, optional, default: 16
        The font size of the title text. Only relevant if `stacked=False`
    rank_ecdf_color   : str, optional, default: '#a34f4f'
        The color to use for the rank ECDFs
    fill_color        : str, optional, default: 'grey'
        The color of the fill arguments.
    **kwargs          : dict, optional, default: {}
        Keyword arguments can be passed to control the behavior of ECDF simultaneous band computation
        through the `ecdf_bands_kwargs` dictionary. See `simultaneous_ecdf_bands` for keyword arguments
    Returns
    -------
    f : plt.Figure - the figure instance for optional saving
    Raises
    ------
    ShapeError
        If there is a deviation form the expected shapes of `post_samples` and `prior_samples`.
    """


    # Store reference to number of parameters
    n_params = post_samples.shape[-1]

    # Compute fractional ranks (using broadcasting)
    ranks = np.sum(post_samples < prior_samples[:, np.newaxis, :], axis=1) / post_samples.shape[1]

    # Prepare figure
    if stacked:
        f, ax = plt.subplots(1, 1, figsize=fig_size)
    else:
        # Determine n_subplots dynamically
        n_row = int(np.ceil(n_params / 6))
        n_col = int(np.ceil(n_params / n_row))

        # Determine fig_size dynamically, if None
        if fig_size is None:
            fig_size = (int(5 * n_col), int(5 * n_row))

        # Initialize figure
        f, ax = plt.subplots(n_row, n_col, figsize=fig_size)

    # Plot individual ecdf of parameters
    for j in range(ranks.shape[-1]):

        ecdf_single = np.sort(ranks[:, j])
        xx = ecdf_single
        yy = np.arange(1, xx.shape[-1] + 1) / float(xx.shape[-1])

        # Difference, if specified
        if difference:
            yy -= xx

        if stacked:
            if j == 0:
                ax.plot(xx, yy, color=rank_ecdf_color, alpha=0.95, label="Rank ECDFs")
            else:
                ax.plot(xx, yy, color=rank_ecdf_color, alpha=0.95)
        else:
            ax.flat[j].plot(xx, yy, color=rank_ecdf_color, alpha=0.95, label="Rank ECDF")

    # Compute uniform ECDF and bands
    alpha, z, L, H = simultaneous_ecdf_bands(post_samples.shape[0], **kwargs.pop("ecdf_bands_kwargs", {}))

    # Difference, if specified
    if difference:
        L -= z
        H -= z

    # Add simultaneous bounds
    if stacked:
        titles = [None]
        axes = [ax]

    else:
        axes = ax.flat
        if param_names is None:
            titles = [f"$p_{{{i}}}$" for i in range(1, n_params + 1)]
        else:
            titles = param_names

    for _ax, title in zip(axes, titles):
        _ax.fill_between(z, L, H, color=fill_color, alpha=0.2, label=rf"{int((1-alpha) * 100)}$\%$ Confidence Bands")

        # Prettify plot
        sns.despine(ax=_ax)
        _ax.grid(alpha=0.35)

        if difference:
            ylab = "ECDF difference"
        else:
            ylab = "ECDF"
        
        _ax.set_title(title, fontsize=title_fontsize)
        _ax.tick_params(axis='both', which='major', labelsize=16)

    # set xlabels
    bottom_row = ax if n_row == 1 else ax[0] if n_col == 1 else ax[n_row-1, :]
    for _ax in bottom_row:
        _ax.set_xlabel("Fractional rank statistic", fontsize=label_fontsize)
    
    # set ylabels
    if n_row == 1:
        ax[0].set_ylabel(ylab, fontsize=label_fontsize)
    else: 
        for _ax in ax[:, 0]:
            _ax.set_ylabel(ylab, fontsize=label_fontsize)
        
    for _ax in axes[n_params:]:
        _ax.remove()
        
    f.tight_layout()
    return f
