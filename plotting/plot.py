"""
Unified plotting module for Experiments 1 and 2
Eliminates code duplication while maintaining experiment-specific functionality
"""

import numpy as np
import pingouin as pg
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import FontProperties
from pathlib import Path
from mne.stats import permutation_cluster_1samp_test
import plotting.PtitPrince as pt
import os


# =====================================================================
# Utility Functions (shared across experiments)
# =====================================================================

def find_nearest(array, value):
    """Find nearest value in array and return index and value."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def get_font_properties(font_path="./fonts/HelveticaWorld-Regular.ttf"):
    """Get font properties for consistent styling."""
    fpath = Path(font_path)
    return FontProperties(fname=fpath)


# =====================================================================
# TFCE Statistics Functions
# =====================================================================

def get_tfce_stats(data, measure='distance', n_perms=10000, experiment=1):
    """
    Compute TFCE statistics for either experiment.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data containing subject, nvoxels, and optionally expected columns
    measure : str
        Either 'distance' or 'correct'
    n_perms : int
        Number of permutations
    experiment : int
        1 or 2 to specify experiment-specific processing
    """
    if experiment == 1:
        # Exp1: Group by subject, nvoxels, and expected
        grouped_data = data.groupby(['subject', 'nvoxels', 'expected']).mean().reset_index()
    else:
        # Exp2: Group by subject and nvoxels only
        grouped_data = data.groupby(['subject', 'nvoxels']).mean().reset_index()
    
    subxvoxels = df_to_array_tfce(grouped_data, measure=measure, experiment=experiment)
    
    # For Exp2 accuracy, subtract chance level
    if experiment == 2 and measure == 'correct':
        subxvoxels -= 0.5
    
    threshold_tfce = dict(start=0, step=0.01)
    t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
        subxvoxels, n_jobs=1, threshold=threshold_tfce, adjacency=None,
        n_permutations=n_perms, out_type='mask')
    
    return t_obs, clusters, cluster_pv, H0


def df_to_array_tfce(df, measure='correct', experiment=1):
    """
    Convert dataframe to array for TFCE analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Grouped dataframe
    measure : str
        Measure to extract ('correct' or 'distance')
    experiment : int
        1 or 2 to specify experiment-specific processing
    """
    subxvoxels = np.zeros((df.subject.nunique(), df.nvoxels.nunique()))
    
    for i, sub in enumerate(np.sort(df.subject.unique())):
        for j, nv in enumerate(np.sort(df.nvoxels.unique())):
            thisdata = df[(df['subject'] == sub) & (df['nvoxels'] == nv)]
            
            if experiment == 1:
                # Exp1: Calculate difference between expected and unexpected
                exp_val = thisdata[thisdata['expected'] == True][measure].values
                unexp_val = thisdata[thisdata['expected'] == False][measure].values
                if len(exp_val) > 0 and len(unexp_val) > 0:
                    subxvoxels[i, j] = exp_val[0] - unexp_val[0]
            else:
                # Exp2: Use raw values
                if len(thisdata) > 0:
                    subxvoxels[i, j] = thisdata[measure].values[0]
    
    return subxvoxels


# =====================================================================
# Data Transformation Functions (Experiment 1 specific)
# =====================================================================

def accs_to_diffs(df, measure='distance'):
    """Convert accuracies to differences between conditions (Exp1 only)."""
    diffs = []
    for nv in df.nvoxels.unique():
        for hemi in df.hemi.unique():
            for sub in df[(df['nvoxels'] == nv) & (df['hemi'] == hemi)].subject.unique():
                thissub = df[(df['nvoxels'] == nv) & (df['hemi'] == hemi) & (df['subject'] == sub)]
                exp_data = thissub[thissub['expected'] == True][measure].values
                unexp_data = thissub[thissub['expected'] == False][measure].values
                if len(exp_data) > 0 and len(unexp_data) > 0:
                    thisdiff = exp_data[0] - unexp_data[0]
                    diffs.append({'subject': sub, 'nvoxels': nv, 'hemi': hemi, 'difference': thisdiff})
    return pd.DataFrame(diffs)


def meanbetas_to_diffs(df):
    """Convert mean betas to differences between conditions (Exp1 only)."""
    diffs = []
    for nv in df.nvoxels.unique():
        for hemi in df.hemi.unique():
            for sub in df[(df['nvoxels'] == nv) & (df['hemi'] == hemi)].subject.unique():
                thissub = df[(df['nvoxels'] == nv) & (df['hemi'] == hemi) & (df['subject'] == sub)]
                exp_data = thissub[thissub['condition'] == 'expected']['mean_beta'].values
                unexp_data = thissub[thissub['condition'] == 'unexpected']['mean_beta'].values
                if len(exp_data) > 0 and len(unexp_data) > 0:
                    thisdiff = exp_data[0] - unexp_data[0]
                    diffs.append({'subject': sub, 'nvoxels': nv, 'hemi': hemi, 'difference': thisdiff})
    return pd.DataFrame(diffs)


# =====================================================================
# Hemisphere Violin Plot Component
# =====================================================================

def draw_hemisphere_violins(ax, data, measure, nsubjs, fontprop, fixed_ylim=True, 
                           plot_difference=True, experiment=1):
    """
    Draw hemisphere comparison violin plots.
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    data : pd.DataFrame
        Data averaged by subject and hemisphere
    measure : str
        'distance' or 'correct'
    nsubjs : int
        Number of subjects
    fontprop : FontProperties
        Font properties for labels
    fixed_ylim : bool
        Whether to use fixed y-limits
    plot_difference : bool
        Whether to plot difference (Exp1) or raw values (Exp2)
    experiment : int
        1 or 2 for experiment-specific settings
    """
    # Draw half violins for each hemisphere
    _, suppL, densL = pt.half_violinplot(
        y='difference' if plot_difference else measure, 
        data=data[data['hemi'] == 'L'], 
        color='.8', width=.3, inner=None, bw=.4, flip=False, CI=True, offset=0.04
    )
    _, suppR, densR = pt.half_violinplot(
        y='difference' if plot_difference else measure,
        data=data[data['hemi'] == 'R'], 
        color='.8', width=.3, inner=None, bw=.4, flip=True, CI=True, offset=0.04
    )
    
    # Draw scatter points with density-based jitter
    target_col = 'difference' if plot_difference else measure
    
    # Left hemisphere
    densities_left = []
    for d in data[data['hemi'] == 'L'][target_col]:
        ix, _ = find_nearest(suppL[0], d)
        densities_left.append(densL[0][ix])
    densities_left = np.array(densities_left).reshape(nsubjs, 1)
    scatter_left = -0.04 - np.random.uniform(size=(nsubjs, 1)) * densities_left * 0.15
    plt.scatter(scatter_left, data[data['hemi'] == 'L'][target_col], color='black', alpha=.3)
    
    # Right hemisphere
    densities_right = []
    for d in data[data['hemi'] == 'R'][target_col]:
        ix, _ = find_nearest(suppR[0], d)
        densities_right.append(densR[0][ix])
    densities_right = np.array(densities_right).reshape(nsubjs, 1)
    scatter_right = 0.04 + np.random.uniform(size=(nsubjs, 1)) * densities_right * 0.15
    plt.scatter(scatter_right, data[data['hemi'] == 'R'][target_col], color='black', alpha=.3)
    
    # Get mean and 95% CI
    if plot_difference:
        mean_val = data['difference'].mean()
        test_data = data.groupby(['subject']).mean().reset_index()['difference']
    else:
        mean_val = data[measure].mean()
        test_data = data.groupby(['subject']).mean().reset_index()[measure]
    
    tstats = pg.ttest(test_data, 0.0)
    ci95 = tstats['CI95%'][0]
    
    # Draw CI and mean
    for tick in ax.get_xticks():
        ax.plot([tick, tick], [ci95[0], ci95[1]], lw=3, color='k')
        ax.plot([tick-0.01, tick+0.01], [ci95[0], ci95[0]], lw=3, color='k')
        ax.plot([tick-0.01, tick+0.01], [ci95[1], ci95[1]], lw=3, color='k')
        ax.plot(tick, mean_val, 'o', markersize=15, color='black')
    
    # Set chance level line
    chance_level = 0.0 if plot_difference or measure == 'distance' else 0.5
    ax.axhline(chance_level, linestyle='--', color='black', linewidth=2)
    
    # Styling
    plt.yticks(font=fontprop.get_name(), fontsize=36)
    ax.set_xticks([-0.07, 0.07])
    ax.set_xticklabels(['Left', 'Right'], font=fontprop.get_name(), fontsize=36)
    plt.xticks(font=fontprop.get_name(), fontsize=36)
    ax.set_xlabel('Hemisphere', font=fontprop.get_name(), fontsize=36, labelpad=16)
    
    # Set y-axis label based on measure and plot type
    if plot_difference:
        ylabel = 'Δ Classifier Information (a.u.)' if measure == 'distance' else 'Δ Decoding Accuracy (a.u.)'
    else:
        ylabel = 'Classifier Information (a.u.)' if measure == 'distance' else 'Decoding Accuracy (a.u.)'
    ax.set_ylabel(ylabel, font=fontprop.get_name(), fontsize=36)
    
    # Set y-limits
    if fixed_ylim:
        if experiment == 1:
            ax.set(ylim=(-0.7, 0.7))
        else:
            ax.set(ylim=(-0.8, 0.8) if measure == 'distance' else (0.1, 0.9))
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(2)


# =====================================================================
# Main Plotting Function
# =====================================================================

def plot_by_nvoxels(data, measure='distance', tfce_pvals=None, right_part=False, 
                   n_perms=10000, fixed_ylim=True, experiment=1):
    """
    Main plotting function for both experiments.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    measure : str
        'distance' or 'correct'
    tfce_pvals : array-like or None
        Pre-computed TFCE p-values
    right_part : bool
        Whether to include hemisphere comparison
    n_perms : int
        Number of permutations for TFCE
    fixed_ylim : bool
        Whether to use fixed y-limits
    experiment : int
        1 or 2 to specify experiment-specific behavior
    """
    fontprop = get_font_properties()
    
    if right_part:
        assert 'hemi' in data.columns
    assert data.roi.nunique() == 1
    
    avgdata = data.copy()
    nsubjs = avgdata.subject.nunique()
    maxvoxels = avgdata.nvoxels.max()
    
    # Compute TFCE stats if not provided
    if tfce_pvals is None:
        if experiment == 1:
            stats_data = avgdata.groupby(['subject', 'nvoxels', 'expected']).mean().reset_index()
        else:
            stats_data = avgdata.groupby(['subject', 'nvoxels']).mean().reset_index()
        _, _, tfce_pvals, _ = get_tfce_stats(stats_data, measure=measure, 
                                            n_perms=n_perms, experiment=experiment)
    
    # Sort n. voxels and make categorical
    avgdata.sort_values('nvoxels', inplace=True, ascending=True)
    avgdata.loc[:, 'nvoxels'] = avgdata.loc[:, 'nvoxels'].astype(str)
    avgdata.loc[:, 'nvoxels'] = pd.Categorical(
        avgdata.loc[:, 'nvoxels'], 
        categories=avgdata.nvoxels.unique(), 
        ordered=True
    )
    
    # Configure plot settings based on experiment and measure
    plot_config = get_plot_config(experiment, measure)
    
    # Create figure
    fig = plt.figure(figsize=(24, 9), facecolor='white')
    gs = GridSpec(4, 4, figure=fig, height_ratios=[0.2, 4.8, 4.8, 0.2])
    
    # Main plot
    with sns.axes_style('white'):
        ax0 = fig.add_subplot(gs[1:3, 1:] if right_part else gs[1:3, :])
        
        # Plot lines based on experiment
        if experiment == 1:
            # Exp1: Plot expected vs unexpected
            plot_data = avgdata.groupby(['subject', 'nvoxels', 'expected']).mean().reset_index()
            sns.lineplot(
                data=plot_data,
                x='nvoxels', y=measure,
                hue='expected', hue_order=[True, False],
                palette='Dark2', ci=68, marker='o', mec='none', markersize=10
            )
            # Fix legend
            ax0.legend_.set_title(None)
            fontprop.set_size(36)
            ax0.legend(['Congruent', 'Incongruent'], prop=fontprop, frameon=False)
        else:
            # Exp2: Single condition
            plot_data = avgdata.groupby(['subject', 'nvoxels']).mean().reset_index()
            sns.lineplot(
                data=plot_data,
                x='nvoxels', y=measure,
                palette='Dark2', ci=95, marker='o', mec='none', markersize=10
            )
        
        # Add chance level line for Exp2
        if experiment == 2:
            ax0.axhline(plot_config['chance_level'], color='k', linestyle='--', linewidth=3.)
        
        # Styling
        plt.yticks(font=fontprop.get_name(), fontsize=36, ticks=plot_config['yticks'])
        ax0.set(ylim=plot_config['ylimits_left'], 
               xticks=['100', '500'] + [str(x) for x in np.arange(1000, maxvoxels+1000, 1000)])
        ax0.set_xlabel('Number of Voxels', font=fontprop.get_name(), fontsize=36)
        ax0.set_ylabel(plot_config['ylabel'], font=fontprop.get_name(), fontsize=36)
        plt.xticks(font=fontprop.get_name(), fontsize=36)
        plt.margins(0.02)
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.spines['left'].set_linewidth(2)
        ax0.spines['bottom'].set_linewidth(2)
        
        # Add significance markers
        add_significance_markers(ax0, tfce_pvals, plot_config, experiment)
    
    # Hemisphere comparison plot (if requested)
    if right_part:
        with sns.axes_style('white'):
            ax1 = fig.add_subplot(gs[:, 0])
            
            if experiment == 1:
                # Exp1: Plot differences
                avgdiffs = accs_to_diffs(avgdata, measure=measure)
                avgdiffs = avgdiffs.groupby(['subject', 'hemi']).mean().reset_index()
                draw_hemisphere_violins(ax1, avgdiffs, measure, nsubjs, fontprop, 
                                      fixed_ylim, plot_difference=True, experiment=1)
            else:
                # Exp2: Plot raw values
                hemi_data = avgdata.groupby(['subject', 'hemi']).mean().reset_index()
                draw_hemisphere_violins(ax1, hemi_data, measure, nsubjs, fontprop,
                                      fixed_ylim, plot_difference=False, experiment=2)
    
    plt.subplots_adjust(wspace=0.4)
    plt.tight_layout()
    
    return fig


def get_plot_config(experiment, measure):
    """Get plot configuration based on experiment and measure."""
    config = {}
    
    if experiment == 1:
        if measure == 'distance':
            config['ylabel'] = 'Classifier Information (a.u.)'
            config['ylimits_left'] = (0.0, 0.45)
            config['yticks'] = list(np.arange(0., 0.45, 0.1))
            config['marker_positions'] = [0.02, 0.04]
            config['chance_level'] = 0.0
        else:  # correct
            config['ylabel'] = 'Decoding Accuracy (a.u.)'
            config['ylimits_left'] = (0.5, 0.75)
            config['yticks'] = list(np.arange(0.5, 0.75, 0.1))
            config['marker_positions'] = [0.52, 0.54]
            config['chance_level'] = 0.5
    else:  # experiment == 2
        if measure == 'distance':
            config['ylabel'] = 'Classifier Information (a.u.)'
            config['ylimits_left'] = (-0.05, 0.2)
            config['ylimits_right'] = (-0.8, 0.8)
            config['yticks'] = list(np.arange(-0.05, 0.2, 0.05))
            config['marker_positions'] = [-0.04, -0.03, -0.01]
            config['chance_level'] = 0.0
        else:  # correct
            config['ylabel'] = 'Decoding Accuracy (a.u.)'
            config['ylimits_left'] = (0.45, 0.6)
            config['ylimits_right'] = (0.1, 0.9)
            config['yticks'] = list(np.arange(0.45, 0.65, 0.05))
            config['marker_positions'] = [0.46, 0.467, 0.474]
            config['chance_level'] = 0.5
    
    return config


def add_significance_markers(ax, tfce_pvals, config, experiment):
    """Add significance markers to the plot."""
    markers = config['marker_positions']
    
    for x in np.arange(0, len(tfce_pvals)):
        if experiment == 1:
            # Exp1: Two levels of significance
            if tfce_pvals[x] < 0.01:
                ax.scatter(x, markers[0], marker=(6, 2, 0), s=180, color='k', linewidths=3.)
                ax.scatter(x, markers[1], marker=(6, 2, 0), s=180, color='k', linewidths=3.)
            elif tfce_pvals[x] < 0.05:
                ax.scatter(x, markers[0], marker=(6, 2, 0), s=180, color='k', linewidths=3.)
        else:
            # Exp2: Three levels of significance
            if tfce_pvals[x] < 0.001:
                for pos in markers:
                    ax.scatter(x, pos, marker=(6, 2, 0), s=180, color='k', linewidths=3.)
            elif tfce_pvals[x] < 0.01:
                for pos in markers[:2]:
                    ax.scatter(x, pos, marker=(6, 2, 0), s=180, color='k', linewidths=3.)
            elif tfce_pvals[x] < 0.05:
                ax.scatter(x, markers[0], marker=(6, 2, 0), s=180, color='k', linewidths=3.)


# =====================================================================
# Experiment 1 Specific Functions
# =====================================================================

def plot_univar_by_nvoxels(data):
    """Plot univariate analysis by nvoxels (Exp1 only)."""
    fontprop = get_font_properties()
    
    avgdata = data.copy()
    nsubjs = avgdata.subject.nunique()
    maxvoxels = avgdata.nvoxels.max()
    
    # Sort n. voxels and make categorical
    avgdata.sort_values('nvoxels', inplace=True, ascending=True)
    avgdata.loc[:, 'nvoxels'] = avgdata.loc[:, 'nvoxels'].astype(str)
    avgdata.loc[:, 'nvoxels'] = pd.Categorical(
        avgdata.loc[:, 'nvoxels'], 
        categories=avgdata.nvoxels.unique(), 
        ordered=True
    )
    
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(4, 4, figure=fig, height_ratios=[1, 4, 4, 1])
    ax0 = fig.add_subplot(gs[1:3, 1:])
    
    with sns.axes_style('white'):
        sns.lineplot(
            data=avgdata.groupby(['subject', 'nvoxels', 'condition']).mean().reset_index(),
            x='nvoxels', y='mean_beta',
            hue='condition', hue_order=['expected', 'unexpected'],
            palette='Dark2', ci=68, marker='o', mec='none', markersize=10
        )
    
    plt.yticks(font=fontprop.get_name(), fontsize=28, ticks=list(np.arange(-6.0, 1.0, 1.0)))
    ax0.set(ylim=(-6.0, 1.0), 
           xticks=['100', '500'] + [str(x) for x in np.arange(1000, maxvoxels+1000, 1000)])
    ax0.set_xlabel('Number of Voxels', font=fontprop.get_name(), fontsize=32)
    ax0.set_ylabel('Mean Beta Value (a.u.)', font=fontprop.get_name(), fontsize=32)
    plt.xticks(font=fontprop.get_name(), fontsize=28)
    plt.margins(0.02)
    ax0.legend_.set_title(None)
    fontprop.set_size(28)
    ax0.legend(['Congruent', 'Incongruent'], prop=fontprop, frameon=False)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['left'].set_linewidth(2)
    ax0.spines['bottom'].set_linewidth(2)
    ax0.axhline(0.0, linestyle='--', color='black', linewidth=2)
    
    # Average violin plot
    avgdiffs = meanbetas_to_diffs(avgdata).groupby(['subject', 'hemi']).mean().reset_index()
    
    with sns.axes_style('white'):
        ax1 = fig.add_subplot(gs[:, 0])
        _, suppL, densL = pt.half_violinplot(
            y='difference', data=avgdiffs[avgdiffs['hemi'] == 'L'], 
            color='.8', width=.3, inner=None, bw=.4, flip=False, CI=True, offset=0.04
        )
        _, suppR, densR = pt.half_violinplot(
            y='difference', data=avgdiffs[avgdiffs['hemi'] == 'R'],
            color='.8', width=.3, inner=None, bw=.4, flip=True, CI=True, offset=0.04
        )
        
        # Draw scatter points
        densities_left = []
        for d in avgdiffs[avgdiffs['hemi'] == 'L']['difference']:
            ix, _ = find_nearest(suppL[0], d)
            densities_left.append(densL[0][ix])
        densities_left = np.array(densities_left).reshape(nsubjs, 1)
        scatter_left = -0.04 - np.random.uniform(size=(nsubjs, 1)) * densities_left * 0.15
        plt.scatter(scatter_left, avgdiffs[avgdiffs['hemi'] == 'L']['difference'], 
                   color='black', alpha=.3)
        
        densities_right = []
        for d in avgdiffs[avgdiffs['hemi'] == 'R']['difference']:
            ix, _ = find_nearest(suppR[0], d)
            densities_right.append(densR[0][ix])
        densities_right = np.array(densities_right).reshape(nsubjs, 1)
        scatter_right = 0.04 + np.random.uniform(size=(nsubjs, 1)) * densities_right * 0.15
        plt.scatter(scatter_right, avgdiffs[avgdiffs['hemi'] == 'R']['difference'], 
                   color='black', alpha=.3)
        
        # Get mean and 95% CI
        meandiff = avgdiffs['difference'].mean()
        tstats = pg.ttest(avgdiffs.groupby(['subject']).mean().reset_index()['difference'], 0.0)
        ci95 = tstats['CI95%'][0]
        
        for tick in ax1.get_xticks():
            ax1.plot([tick, tick], [ci95[0], ci95[1]], lw=3, color='k')
            ax1.plot([tick-0.01, tick+0.01], [ci95[0], ci95[0]], lw=3, color='k')
            ax1.plot([tick-0.01, tick+0.01], [ci95[1], ci95[1]], lw=3, color='k')
            ax1.plot(tick, meandiff, 'o', markersize=15, color='black')
        
        ax1.axhline(0.0, linestyle='--', color='black', linewidth=2)
        plt.yticks(font=fontprop.get_name(), fontsize=32)
        ax1.set_xlabel('Average', font=fontprop.get_name(), fontsize=32)
        ax1.set_ylabel('Δ Mean Beta (a.u.)', font=fontprop.get_name(), fontsize=32)
        ax1.set(ylim=(-2.5, 2.5))
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_linewidth(2)
    
    plt.subplots_adjust(wspace=0.4)
    plt.tight_layout()
    
    return fig


def pretty_behav_plot(avgdata, measure='Hit', excl=True, fname=None, saveimg=False):
    """Create behavioral plot (Exp1 only)."""
    assert measure in ['Hit', 'DPrime', 'Criterion']
    
    fontprop = get_font_properties()
    
    # Get all differences
    alldiffs = []
    for sub in avgdata.Subject.unique():
        cons_data = avgdata[(avgdata.Subject == sub) & (avgdata.Consistent == 1)][measure].values
        incons_data = avgdata[(avgdata.Subject == sub) & (avgdata.Consistent == 0)][measure].values
        if len(cons_data) > 0 and len(incons_data) > 0:
            thisdiff = cons_data[0] - incons_data[0]
            alldiffs.append(thisdiff)
    alldiffs = pd.DataFrame(alldiffs, columns=['difference'])
    
    fig = plt.figure(figsize=(10, 10))
    
    # Bar plot
    ax0 = fig.add_subplot(121)
    sns.barplot(
        x='Consistent', y=measure, data=avgdata, ci=68, order=[1.0, 0.0],
        palette='Set2', ax=ax0, errcolor='black', edgecolor='black', 
        linewidth=2, capsize=.2
    )
    
    # Set labels based on measure
    ylabel_map = {'Hit': 'Accuracy', 'DPrime': "d'", 'Criterion': 'Criterion'}
    ax0.set_ylabel(ylabel_map[measure], font=fontprop.get_name(), fontsize=34)
    
    plt.yticks(font=fontprop.get_name(), fontsize=28)
    ax0.tick_params(axis='y', direction='out', color='black', length=10, width=2)
    ax0.tick_params(axis='x', length=0, pad=15)
    ax0.set_xlabel(None)
    ax0.set_xticklabels(['Cong.', 'Incong.'], font=fontprop.get_name(), fontsize=34)
    ax0.spines['left'].set_linewidth(2)
    ax0.spines['bottom'].set_linewidth(2)
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    
    # Set y-limits based on measure
    ylim_map = {
        'Hit': (0.5, 0.75),
        'DPrime': (0.0, 1.0),
        'Criterion': (0.0, 1.0)
    }
    ax0.set(ylim=ylim_map[measure])
    
    # Difference plot
    ax1 = fig.add_subplot(122)
    sns.violinplot(y='difference', data=alldiffs, color=".8", inner=None)
    sns.stripplot(y='difference', data=alldiffs, jitter=0.07, ax=ax1, color='black', alpha=.5)
    
    # Get mean and 95% CI
    meandiff = alldiffs['difference'].mean()
    tstats = pg.ttest(alldiffs['difference'], 0.0)
    ci95 = tstats['CI95%'][0]
    
    for tick in ax1.get_xticks():
        ax1.plot([tick-0.1, tick+0.1], [meandiff, meandiff], lw=4, color='k')
        ax1.plot([tick, tick], [ci95[0], ci95[1]], lw=3, color='k')
        ax1.plot([tick-0.03, tick+0.03], [ci95[0], ci95[0]], lw=3, color='k')
        ax1.plot([tick-0.03, tick+0.03], [ci95[1], ci95[1]], lw=3, color='k')
    
    ax1.axhline(0.0, linestyle='--', color='black')
    plt.yticks(font=fontprop.get_name(), fontsize=28)
    
    # Set difference label
    diff_ylabel_map = {'Hit': 'Δ Accuracy', 'DPrime': "Δ d'", 'Criterion': 'Δ Criterion'}
    ax1.set_ylabel(diff_ylabel_map[measure], font=fontprop.get_name(), fontsize=34)
    
    # Set y-limits for difference plot
    diff_ylim_map = {
        'Hit': (-0.2, 0.4) if excl else (-0.3, 0.4),
        'DPrime': (-2., 2.),
        'Criterion': (-1.0, 1.25)
    }
    ax1.set(ylim=diff_ylim_map[measure])
    
    ax1.tick_params(axis='y', direction='out', color='black', length=10, width=2)
    ax1.tick_params(axis='x', length=0)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    
    plt.tight_layout()
    
    if saveimg:
        if not fname:
            fname = f'behavior_{measure}.svg'
            if not excl:
                fname = fname.replace('.svg', '_noexcl.svg')
        if not os.path.isdir('results_plots'):
            os.mkdir('results_plots')
        plt.savefig(os.path.join('results_plots', fname))
    
    return fig