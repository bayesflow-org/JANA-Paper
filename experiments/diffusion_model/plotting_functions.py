from scipy.stats import binom, median_abs_deviation
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import sys
import numpy as np
import pandas as pd
import seaborn as sns

import logging
logging.basicConfig()

from bayesflow.helper_functions import check_posterior_prior_shapes
from bayesflow.computational_utilities import simultaneous_ecdf_bands


# adapted from BayesFlow's diagnostics.plot_recovery
def compare_estimates(samples_x, samples_y, point_agg=np.median, uncertainty_agg=np.std, 
                      param_names=None, fig_size=None, label_x='x', label_y='y',
                      label_fontsize=14, title_fontsize=16,
                      metric_fontsize=16, add_corr=True, add_r2=True, color='#8f2727', 
                      markersize=6.0, n_col=None, n_row=None):
    
    """ Creates and plots publication-ready plot with point estimates + uncertainty.
    The point estimates can be controlled with the `point_agg` argument, and the uncertainty estimates
    can be controlled with the `uncertainty_agg` argument.
    Important: Posterior aggregates play no special role in Bayesian inference and should only
    be used heuristically. For instanec, in the case of multi-modal posteriors, common point
    estimates, such as mean, (geometric) median, or maximum a posteriori (MAP) mean nothing.
    Parameters
    ----------
    samples_x         : np.ndarray of shape (n_data_sets, n_post_draws, n_params)
        The first set of posterior draws obtained from n_data_sets
    samples_y         : np.ndarray of shape (n_data_sets, n_post_draws, n_params)
        The second set of posterior draws obtained from n_data_sets
    point_agg         : callable, optional, default: np.mean
        The function to apply to the posterior draws to get a point estimate for each marginal.
    uncertainty_agg   : callable or None, optional, default: np.std
        The function to apply to the posterior draws to get an uncertainty estimate.
        If `None` provided, a simple scatter will be plotted.
    param_names       : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    fig_size          : tuple or None, optional, default : None
        The figure size passed to the matplotlib constructor. Inferred if None.
    label_x           : string, optional, default: 'x'
        The x-label text
    label_y           : string, optional, default: 'y'
        The y-label text
    label_fontsize    : int, optional, default: 14
        The font size of the y-label text
    title_fontsize    : int, optional, default: 16
        The font size of the title text
    metric_fontsize   : int, optional, default: 16
        The font size of the goodness-of-fit metric (if provided)
    add_corr          : boolean, optional, default: True
        A flag for adding correlation between true and estimates to the plot.
    add_r2            : boolean, optional, default: True
        A flag for adding R^2 between true and estimates to the plot.
    color             : str, optional, default: '#8f2727'
        The color for the true vs. estimated scatter points and errobars.
    markersize        : float, optional, default: 6.0
        The marker size in points.
        
    Returns
    -------
    f : plt.Figure - the figure instance for optional saving
    """
    
    # Compute point estimates and uncertainties
    est_x = point_agg(samples_x, axis=1)
    est_y = point_agg(samples_y, axis=1)
    if uncertainty_agg is not None:
        u_x = uncertainty_agg(samples_x, axis=1)
        u_y = uncertainty_agg(samples_y, axis=1)
    
    # Determine n params and param names if None given
    n_params = samples_x.shape[-1]
    if param_names is None:
        param_names = [f'p_{i}' for i in range(1, n_params+1)]
        
    # Determine number of rows and columns for subplots based on inputs
    if n_row is None and n_col is None:
        n_row = int(np.ceil(n_params / 6))
        n_col = int(np.ceil(n_params / n_row))
    elif n_row is None and n_col is not None:
        n_row = int(np.ceil(n_params / n_col))
    elif n_row is not None and n_col is None:
        n_col = int(np.ceil(n_params / n_row))
        
    
    # Initialize figure
    if fig_size is None:
        fig_size = (int(4 * n_col), int(4 * n_row))
    f, axarr = plt.subplots(n_row, n_col, figsize=fig_size)

    # turn axarr into 1D list
    if n_col > 1 or n_row > 1:
        axarr = axarr.flat
    else:
        # for 1x1, axarr is not a list -> turn it into one for use with enumerate
        axarr = [axarr]

    for i, ax in enumerate(axarr):
        if i >= n_params:
            break

        # Add scatter and errorbars
        if uncertainty_agg is not None:
            if len(u_x.shape) == 3:
                im = ax.errorbar(est_x[:, i], est_y[:, i], xerr=u_x[:, :, i], yerr=u_y[:, :, i], fmt='o', alpha=0.5, color=color, markersize=markersize)
            else:
                im = ax.errorbar(est_x[:, i], est_y[:, i], xerr=u_x[:, i], yerr=u_y[:, i], fmt='o', alpha=0.5, color=color, markersize=markersize)
        else:
            im = ax.scatter(est_x[:, i], est_y[:, i], alpha=0.5, color=color, s=markersize**2)

        # Make plots quadratic to avoid visual illusions
        lower = min(est_x[:, i].min(), est_y[:, i].min())
        upper = max(est_x[:, i].max(), est_y[:, i].max())
        eps = (upper - lower) * 0.1
        ax.set_xlim([lower - eps, upper + eps])
        ax.set_ylim([lower - eps, upper + eps]) 
        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [ax.get_ylim()[0], ax.get_ylim()[1]], 
                 color='black', alpha=0.9, linestyle='dashed')
        
        # Add labels, optional metrics and title
        ax.set_xlabel(label_x, fontsize=label_fontsize)
        ax.set_ylabel(label_y, fontsize=label_fontsize)
        if add_r2:
            r2 = r2_score(est_x[:, i], est_y[:, i])
            ax.text(0.1, 0.9, '$R^2$ = {:.3f}'.format(r2),
                     horizontalalignment='left',
                     verticalalignment='center',
                     transform=ax.transAxes, 
                     size=metric_fontsize)
        if add_corr:
            corr = np.corrcoef(est_x[:, i], est_y[:, i])[0, 1]
            ax.text(0.1, 0.8, '$r$ = {:.3f}'.format(corr),
                         horizontalalignment='left',
                         verticalalignment='center',
                         transform=ax.transAxes, 
                         size=metric_fontsize)
        ax.set_title(param_names[i], fontsize=title_fontsize)
        
        # Prettify
        sns.despine(ax=ax)
        ax.grid(alpha=0.5)
    f.tight_layout()
    return f


# Adapted from BayesFlow diagnostic.plot_recovery
# to allow for asymmetric uncertainty aggregation functions
def plot_recovery(post_samples, prior_samples, point_agg=np.median, uncertainty_agg=median_abs_deviation,
                  param_names=None, fig_size=None, label_fontsize=14, title_fontsize=16,
                  metric_fontsize=16, add_corr=True, add_r2=True, color='#8f2727', 
                  n_col=None, n_row=None):
    
    """Creates and plots publication-ready recovery plot with true vs. point estimate + uncertainty.
    The point estimate can be controlled with the ``point_agg`` argument, and the uncertainty estimate
    can be controlled with the ``uncertainty_agg`` argument.
    This plot yields the same information as the "posterior z-score":
    https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html
    Important: Posterior aggregates play no special role in Bayesian inference and should only
    be used heuristically. For instanec, in the case of multi-modal posteriors, common point
    estimates, such as mean, (geometric) median, or maximum a posteriori (MAP) mean nothing.
    Parameters
    ----------
    post_samples      : np.ndarray of shape (n_data_sets, n_post_draws, n_params)
        The posterior draws obtained from n_data_sets
    prior_samples     : np.ndarray of shape (n_data_sets, n_params)
        The prior draws (true parameters) obtained for generating the n_data_sets
    point_agg         : callable, optional, default: np.median
        The function to apply to the posterior draws to get a point estimate for each marginal.
        The default computes the marginal median for each marginal posterior as a robust
        point estimate.
    uncertainty_agg   : callable or None, optional, default: scipy.stats.median_abs_deviation
        The function to apply to the posterior draws to get an uncertainty estimate.
        If ``None`` provided, a simple scatter using only ``point_agg`` will be plotted.
    param_names       : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    fig_size          : tuple or None, optional, default : None
        The figure size passed to the matplotlib constructor. Inferred if None.
    label_fontsize    : int, optional, default: 14
        The font size of the y-label text
    title_fontsize    : int, optional, default: 16
        The font size of the title text
    metric_fontsize   : int, optional, default: 16
        The font size of the goodness-of-fit metric (if provided)
    add_corr          : bool, optional, default: True
        A flag for adding correlation between true and estimates to the plot
    add_r2            : bool, optional, default: True
        A flag for adding R^2 between true and estimates to the plot
    color             : str, optional, default: '#8f2727'
        The color for the true vs. estimated scatter points and errobars
        
    Returns
    -------
    f : plt.Figure - the figure instance for optional saving
    Raises
    ------
    ShapeError 
        If there is a deviation form the expected shapes of `post_samples` and `prior_samples`.
    """

    # Sanity check
    check_posterior_prior_shapes(post_samples, prior_samples)
    
    # Compute point estimates and uncertainties
    est = point_agg(post_samples, axis=1)
    if uncertainty_agg is not None:
        u = uncertainty_agg(post_samples, axis=1)
    
    # Determine n params and param names if None given
    n_params = prior_samples.shape[-1]
    if param_names is None:
        param_names = [f'$p_{i}$' for i in range(1, n_params+1)]
        
    # Determine number of rows and columns for subplots based on inputs
    if n_row is None and n_col is None:
        n_row = int(np.ceil(n_params / 6))
        n_col = int(np.ceil(n_params / n_row))
    elif n_row is None and n_col is not None:
        n_row = int(np.ceil(n_params / n_col))
    elif n_row is not None and n_col is None:
        n_col = int(np.ceil(n_params / n_row))
        
    
    # Initialize figure
    if fig_size is None:
        fig_size = (int(4 * n_col), int(4 * n_row))
    f, axarr = plt.subplots(n_row, n_col, figsize=fig_size)
    # turn axarr into 1D list
    if n_col > 1 or n_row > 1:
        axarr = axarr.flat
    else:
        # for 1x1, axarr is not a list -> turn it into one for use with enumerate
        axarr = [axarr]

    for i, ax in enumerate(axarr):
        if i >= n_params:
            break

        # Add scatter and errorbars
        if uncertainty_agg is not None:
            if len(u.shape) == 3:
                # asymmetric uncertainty estimate (e.g. quantiles)
                im = ax.errorbar(prior_samples[:, i], est[:, i], yerr=u[:, :, i], fmt='o', alpha=0.5, color=color)
            else:
                im = ax.errorbar(prior_samples[:, i], est[:, i], yerr=u[:, i], fmt='o', alpha=0.5, color=color)
        else:
            im = ax.scatter(prior_samples[:, i], est[:, i], alpha=0.5, color=color)

        # Make plots quadratic to avoid visual illusions
        lower = min(prior_samples[:, i].min(), est[:, i].min())
        upper = max(prior_samples[:, i].max(), est[:, i].max())
        eps = (upper - lower) * 0.1
        ax.set_xlim([lower - eps, upper + eps])
        ax.set_ylim([lower - eps, upper + eps]) 
        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [ax.get_ylim()[0], ax.get_ylim()[1]], 
                 color='black', alpha=0.9, linestyle='dashed')
        
        # Add labels, optional metrics and title
        ax.set_xlabel('Ground truth', fontsize=label_fontsize)
        ax.set_ylabel('Estimated', fontsize=label_fontsize)
        if add_r2:
            r2 = r2_score(prior_samples[:, i], est[:, i])
            ax.text(0.1, 0.9, '$R^2$ = {:.3f}'.format(r2),
                     horizontalalignment='left',
                     verticalalignment='center',
                     transform=ax.transAxes, 
                     size=metric_fontsize)
        if add_corr:
            corr = np.corrcoef(prior_samples[:, i], est[:, i])[0, 1]
            ax.text(0.1, 0.8, '$r$ = {:.3f}'.format(corr),
                         horizontalalignment='left',
                         verticalalignment='center',
                         transform=ax.transAxes, 
                         size=metric_fontsize)
        ax.set_title(param_names[i], fontsize=title_fontsize)
        
        # Prettify
        sns.despine(ax=ax)
        ax.grid(alpha=0.5)
    f.tight_layout()
    return f


# Adapted from BayesFlow diagnostic.plot_recovery
# to allow for manual specification of error bars
def compare_point_estimates(est_x, est_y, u_x=None, u_y=None,
                      param_names=None, fig_size=None, label_x='x', label_y='y',
                      label_fontsize=14, title_fontsize=16,
                      metric_fontsize=16, add_corr=True, add_r2=True, color='#8f2727', 
                      markersize=6.0, n_col=None, n_row=None):
    
    """
    Creates and plots publication-ready comparison plots. The uncertainty can be
    provided manually.
    ----------
    est_x             : np.ndarray of shape (n_data_sets, n_params)
        The first set of n_data_sets estimates
    est_y             : np.ndarray of shape (n_data_sets, n_params)
        The second set of n_data_sets estimates
    u_x               : The uncertainty for the first set of estimates
    u_y               : The uncertainty for the second set of estimates
    param_names       : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    fig_size          : tuple or None, optional, default : None
        The figure size passed to the matplotlib constructor. Inferred if None.
    label_x           : string, optional, default: 'x'
        The x-label text
    label_y           : string, optional, default: 'y'
        The y-label text
    label_fontsize    : int, optional, default: 14
        The font size of the y-label text
    title_fontsize    : int, optional, default: 16
        The font size of the title text
    metric_fontsize   : int, optional, default: 16
        The font size of the goodness-of-fit metric (if provided)
    add_corr          : boolean, optional, default: True
        A flag for adding correlation between true and estimates to the plot.
    add_r2            : boolean, optional, default: True
        A flag for adding R^2 between true and estimates to the plot.
    color             : str, optional, default: '#8f2727'
        The color for the true vs. estimated scatter points and errobars.
    markersize        : float, optional, default: 6.0
        The marker size in points.
        
    Returns
    -------
    f : plt.Figure - the figure instance for optional saving
    """
    
    # Determine n params and param names if None given
    n_params = est_x.shape[-1]
    if param_names is None:
        param_names = [f'p_{i}' for i in range(1, n_params+1)]
        
    # Determine number of rows and columns for subplots based on inputs
    if n_row is None and n_col is None:
        n_row = int(np.ceil(n_params / 6))
        n_col = int(np.ceil(n_params / n_row))
    elif n_row is None and n_col is not None:
        n_row = int(np.ceil(n_params / n_col))
    elif n_row is not None and n_col is None:
        n_col = int(np.ceil(n_params / n_row))
        
    
    # Initialize figure
    if fig_size is None:
        fig_size = (int(4 * n_col), int(4 * n_row))
    f, axarr = plt.subplots(n_row, n_col, figsize=fig_size)

	# turn axarr into 1D list
    if n_col > 1 or n_row > 1:
        axarr = axarr.flat
    else:
        # for 1x1, axarr is not a list -> turn it into one for use with enumerate
        axarr = [axarr]

    for i, ax in enumerate(axarr):
        if i >= n_params:
            break

        # Add scatter and errorbars
        if u_x is not None or u_y is not None:
            u_x_i = None
            u_y_i = None
            if u_x is not None:
                u_x_i = u_x[:, :, i] if len(u_x.shape) == 3 else u_x[:, i]
            if u_y is not None:
                u_y_i = u_y[:, :, i] if len(u_y.shape) == 3 else u_y[:, i]
            im = ax.errorbar(est_x[:, i], est_y[:, i], xerr=u_x_i, yerr=u_y_i, fmt='o',
                             alpha=0.5, color=color, markersize=markersize)
        else:
            im = ax.scatter(est_x[:, i], est_y[:, i], alpha=0.5, color=color, s=markersize**2)

        # Make plots quadratic to avoid visual illusions
        lower = min(est_x[:, i].min(), est_y[:, i].min())
        upper = max(est_x[:, i].max(), est_y[:, i].max())
        eps = (upper - lower) * 0.1
        ax.set_xlim([lower - eps, upper + eps])
        ax.set_ylim([lower - eps, upper + eps]) 
        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [ax.get_ylim()[0], ax.get_ylim()[1]], 
                 color='black', alpha=0.9, linestyle='dashed')
        
        # Add labels, optional metrics and title
        ax.set_xlabel(label_x, fontsize=label_fontsize)
        ax.set_ylabel(label_y, fontsize=label_fontsize)
        if add_r2:
            r2 = r2_score(est_x[:, i], est_y[:, i])
            ax.text(0.1, 0.9, '$R^2$ = {:.3f}'.format(r2),
                     horizontalalignment='left',
                     verticalalignment='center',
                     transform=ax.transAxes, 
                     size=metric_fontsize)
        if add_corr:
            corr = np.corrcoef(est_x[:, i], est_y[:, i])[0, 1]
            ax.text(0.1, 0.8, '$r$ = {:.3f}'.format(corr),
                         horizontalalignment='left',
                         verticalalignment='center',
                         transform=ax.transAxes, 
                         size=metric_fontsize)
        ax.set_title(param_names[i], fontsize=title_fontsize)
        
        # Prettify
        sns.despine(ax=ax)
        ax.grid(alpha=0.5)
    f.tight_layout()
    return f


# adapted from experiments/benchmarks/custom_plots.py plot_sbc_ecdf
def plot_sbc_ecdf_custom(
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
        _ax.legend(fontsize=legend_fontsize, loc='lower center', ncol=len(rank_ecdf_colors), handles=patches, 
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
