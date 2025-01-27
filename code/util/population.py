import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from statannotations.Annotator import Annotator


def _draw_variable_trial_back(df, beta_name, trials_back, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(7, 3), constrained_layout=True) 
        
    gs = ax._subplotspec.subgridspec(1, 2, wspace=0.5, width_ratios=(1, 0.8))
    axes = []
    axes.append(ax.get_figure().add_subplot(gs[0]))
    axes.append(ax.get_figure().add_subplot(gs[1]))
        
    plot_range = (-1.0, 1.0) if beta_name == 'bias' else (0, 1.5)

    # df_beta = foraging_analysis_and_export.SessionLogisticRegression & (lab.WaterRestriction & 'water_restriction_number = "XY_10"') & 'beta = "C"' & 'trials_back = 1' & 'session > 32'
    # df_beta = pd.DataFrame(df_beta.fetch())

    q_para = f'beta == "{beta_name}" and trials_back == {trials_back}'
    df_beta = df.query(q_para)
    var_name = rf'$\beta$_{beta_name} (N - {trials_back})'

    # --- 2-d plot with session-wise CI ---
    axes[0].errorbar(x=df_beta.query("trial_group == 'ctrl'")['mean'], 
                 y=df_beta.query("trial_group == 'photostim'")['mean'],
                 xerr=np.abs(np.array(list(df_beta.query("trial_group == 'ctrl'")['error_bar'])).T),
                 yerr=np.abs(np.array(list(df_beta.query("trial_group == 'photostim'")['error_bar'])).T),
                 marker='o', color='k', linestyle='')

    plot_range = [min(axes[0].get_xlim()[0], axes[0].get_ylim()[0]),
                  max(axes[0].get_xlim()[1], axes[0].get_ylim()[1])]
    axes[0].plot(plot_range, plot_range, 'k--')
    
    axes[0].set(#xlim=plot_range, ylim=plot_range, 
                title=var_name, 
                xlabel='Control trials', ylabel='Photostim + 0')
    axes[0].set_aspect('equal', 'box')


    #--- Wilcoxon plot ---
    # with sns.plotting_context("notebook", font_scale=1):
    plotting_parameters = {
    'data': df_beta,
    'x': 'trial_group',
    'y': 'mean',
    }

    axes[1].plot(np.arange(len(df_beta['trial_group'].unique())), 
                 np.matrix([df_beta.query(f'trial_group == "{group}"')['mean']
                            for group in df_beta['trial_group'].unique()]),
                 'ok-', alpha=0.2)
    sns.pointplot(ax=axes[1], 
                  **plotting_parameters,
                  errorbar=('ci', 95), capsize=.1,
                  color='k')
    
    # Add stat annotations
    pairs = [('ctrl', 'photostim'), 
             ('ctrl', 'photostim_next'), 
             ('ctrl', 'photostim_next5')]
    pvalues = [scipy.stats.wilcoxon(x=df_beta.query(f"trial_group == '{x}'")['mean'], 
                                    y=df_beta.query(f"trial_group == '{y}'")['mean']).pvalue
               for x, y in pairs]

    sig = lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    formatted_pvalues = [f"{sig(p)} p={p:.2g}" for p in pvalues]

    annotator = Annotator(axes[1], pairs, **plotting_parameters)
    annotator.set_pvalues(pvalues)
    # annotator.configure(#text_format='simple', 
    #     verbose=False)
    annotator.set_custom_annotations(formatted_pvalues)
    annotator.annotate()
    
    axes[1].set_xticklabels(['Ctrl', 'Stim + 0', '+ 1', '+ 5'], rotation=0, ha='center')
    axes[1].set(#ylim=plot_range, title=var_name, 
                xlabel='', ylabel='Session mean $\pm$ CI')
    axes[1].spines[['right', 'top']].set_visible(False)
    axes[1].set_title(f"{len(df_beta['subject_alias'].unique())} mice, {len(df_beta['session'].unique())} sessions")
        
    ax.remove()
    
    return axes





def _draw_variable_trial_back_linear_reg(df, beta_name, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(7, 3), constrained_layout=True) 
               
    plot_range = (-1.0, 1.0) if beta_name == 'bias' else (0, 1.5)

    q_para = f'variable == "{beta_name}"' if 'reward' not in beta_name else f'variable == "reward" and trials_back == {beta_name.split("_")[-1]}'
    
    df_beta = df.query(q_para)

    #--- Wilcoxon plot ---
    # with sns.plotting_context("notebook", font_scale=1):
    plotting_parameters = {
    'data': df_beta,
    'x': 'trial_group',
    'y': 'beta',
    }

    ax.plot(np.arange(len(df_beta['trial_group'].unique())), 
                 np.matrix([df_beta.query(f'trial_group == "{group}"')['beta']
                            for group in df_beta['trial_group'].unique()]),
                 'ok-', alpha=0.2)
    
    sns.pointplot(ax=ax, 
                  **plotting_parameters,
                  errorbar=('ci', 95), 
                #   errorbar='se',
                  capsize=.1,
                  color='k')
    
    # Add stat annotations
    pairs = [('ctrl', 'photostim'), 
             ('ctrl', 'photostim_next'), 
            ]
    pvalues = [scipy.stats.wilcoxon(x=df_beta.query(f"trial_group == '{x}'")['beta'], 
                                    y=df_beta.query(f"trial_group == '{y}'")['beta']).pvalue
               for x, y in pairs]
    
    sig = lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    formatted_pvalues = [f"{sig(p)} p={p:.2g}" for p in pvalues]

    annotator = Annotator(ax, pairs, **plotting_parameters)
    annotator.set_pvalues(pvalues)
    # annotator.configure(#text_format='star', 
    #                     text_format='simple', 
    #                     verbose=True)
    annotator.set_custom_annotations(formatted_pvalues)
    annotator.annotate()
    
    ax.set_xticklabels(['Ctrl', 'Stim + 0', '+ 1'], rotation=0, ha='center')
    ax.set(#ylim=plot_range, 
           xlabel='', ylabel='Session mean $\pm$ CI')
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_title(f"{beta_name}, {len(df_beta['subject_alias'].unique())} mice, {len(df_beta['session'].unique())} sessions")
    ax.axhline(y=0, c='k', ls='--', lw=1)
    
    if 'reward' in beta_name: ax.invert_yaxis()
            
    return ax