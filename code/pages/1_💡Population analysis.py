import streamlit as st

import importlib
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

from streamlit_util import filter_dataframe, aggrid_interactive_table_session, add_session_filter
from population_util import _draw_variable_trial_back, _draw_variable_trial_back_linear_reg
import seaborn as sns

st.session_state.use_s3 = True


def app():

    with st.sidebar:
        add_session_filter()
        
        
    # st.dataframe(st.session_state.df['logistic_regression'])
    
    
    df_to_do_population = st.session_state.df_session_filtered
    df_log_reg_to_do = df_to_do_population.merge(st.session_state.df['logistic_regression'], on=('subject_id', 'session'), how='inner')
    df_lin_reg_rt_to_do = df_to_do_population.merge(st.session_state.df['linear_regression_rt'], on=('subject_id', 'session'), how='inner')

    cols = st.columns([1, 3])
    
    with cols[0]:    
        if st.checkbox('Plot logistic regression (non-photostim)', True):        
            fig = plot_logistic_regression_non_photostim(df_log_reg_to_do.query('trial_group == "all_no_stim"'))
            if fig is not None:
                buf = BytesIO()
                fig.savefig(buf, format="png")
                st.image(buf, use_column_width=True)

            
        if st.checkbox('Plot linear regression on RT (non-photostim)', True):
            fig = plot_linear_regression_rt_non_photostim(df_lin_reg_rt_to_do.query('trial_group == "all_no_stim"'))
            
            if fig is not None:
                buf = BytesIO()
                fig.savefig(buf, format="png")
                st.image(buf, use_column_width=True)

    with cols[1]:
        if st.checkbox('Plot logistic regression (⚡photostim sessions)', False):
            with st.columns([1, 5])[0]:
                beta_names = st.multiselect('beta names', ['RewC', 'UnrC', 'C'], ['RewC', 'UnrC', 'C'])
                max_trials_back = st.slider('max trials back', 1, 5, 3)
                
            fig = plot_logistic_regression_photostim(df_log_reg_to_do.query('trial_group != "all_no_stim"'), 
                                                    beta_names=beta_names, past_trials_to_plot=range(1, max_trials_back + 1))
            if fig is not None:
                buf = BytesIO()
                fig.savefig(buf, format="png")
                st.image(buf, width=3000)
        
        if st.checkbox('Plot linear regression on RT (⚡photostim sessions)', False):
            fig = plot_linear_regression_rt_photostim(df_lin_reg_rt_to_do.query('trial_group != "all_no_stim"'), 
                                                    beta_names=['reward_1', 'reward_2', 'previous_iti', 'trial_number', 'this_choice', 'constant'])
            if fig is not None:
                buf = BytesIO()
                fig.savefig(buf, format="png")
                st.image(buf)
            


    
    # st.pyplot(fig)
    

@st.cache_data(ttl=3600*24)
def plot_logistic_regression_photostim(df_all, beta_names=['RewC', 'UnrC', 'C'], past_trials_to_plot=(1, 2, 3)):
    if not len(df_all): return None
    
    df_all['error_bar'] = [[x, y] for x, y in zip(df_all['mean'] - df_all.lower_ci,  df_all.upper_ci - df_all['mean'])]
    df_all['h2o_session'] = df_all[['h2o', 'session']].astype(str).apply('_'.join, axis=1)
    
    df_all.query('trial_group != "all_no_stim"', inplace=True)


    fig, axes = plt.subplots(len(past_trials_to_plot), len(beta_names) + 1, 
                            figsize=(10 * (len(beta_names) + 1), 5 * len(past_trials_to_plot)), constrained_layout=False,
                            dpi=200,
                            gridspec_kw=dict(hspace=0.3, wspace=0.3, top=0.9, bottom=0.1, left=0.1, right=0.9),
                           )
    axes = np.atleast_2d(axes)

    for i, trials_back in enumerate(past_trials_to_plot):
        for j, name in enumerate(beta_names):        
            _draw_variable_trial_back(df_all, name, trials_back, ax=axes[i, j])
    _draw_variable_trial_back(df_all, 'bias', 0, ax=axes[0, j + 1])

    for i in range(1, len(past_trials_to_plot)): axes[i, -1].remove()
    
    return fig


def plot_logistic_regression_non_photostim(df_all, max_trials_back=10, ax=None):
    if not len(df_all): return None
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=200,
                               gridspec_kw=dict(bottom=0.2, top=0.9))
        
    xx = np.arange(1, max_trials_back + 1)
    
    plot_spec = {'RewC': ('tab:green', 'reward choices'), 
                 'UnrC': ('tab:red', 'unrewarded choices'), 
                 'C': ('tab:blue', 'choices'), 
                 'bias': ('k', 'right bias')}    

    for name, (col, label) in plot_spec.items():
        
        means = [df_all.query(f'beta == "{name}" and trials_back == {t}')['mean'].mean() 
                 for t in ([0] if name == "bias" else xx)]
        cis = [df_all.query(f'beta == "{name}" and trials_back == {t}')['mean'].sem() * 1.96 
               for t in ([0] if name == "bias" else xx)]
                
        ax.errorbar(x=1 if name == 'bias' else xx,
                    y=means,
                    yerr=cis,
                    ls='-', 
                    marker='o',
                    color=col, 
                    capsize=5, markeredgewidth=1,
                    label=label + ' $\pm$95% CI')
    
    ax.legend()
    ax.set(xlabel='Past trials', ylabel='Logistic regression coeffs')
    ax.axhline(y=0, color='k', linestyle=':', linewidth=0.5)
    ax.set(xticks=[1, 5, 10], ylim=(-0.1, 1.3))

    sns.despine(trim=True)

    n_mice = len(df_all['h2o'].unique())
    n_sessions = len(df_all.groupby(['h2o', 'session']).count())
    fig.suptitle(f'Logistic regression on choice ({n_mice} mice, {n_sessions} sessions)')

    
    return fig



# @st.cache_data(ttl=3600*24)
def plot_linear_regression_rt_photostim(df_all, beta_names):
    if not len(df_all): return None

    fig, axes = plt.subplots(2, 3, 
                            figsize=(15, 10), constrained_layout=False,
                            dpi=200,
                            gridspec_kw=dict(hspace=0.3, wspace=0.3, top=0.9, bottom=0.1, left=0.1, right=0.9),
                           )
    axes = np.atleast_2d(axes)

    for j, beta_name in enumerate(beta_names):        
        _draw_variable_trial_back_linear_reg(df_all, beta_name, ax=axes.flat[j])
    
    return fig



def plot_linear_regression_rt_non_photostim(df_all, ax=None):
    '''
    plot list of linear regressions
    '''
    if not len(df_all): return None

    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=200,
                               gridspec_kw=dict(bottom=0.2, left=0.2))
        
    gs = ax._subplotspec.subgridspec(1, 2, width_ratios=[1, 3], wspace=0.5)
    ax_others = ax.get_figure().add_subplot(gs[0, 0])
    ax_reward = ax.get_figure().add_subplot(gs[0, 1])
    
    # Other paras
    other_names = ['previous_iti', 'this_choice', 'trial_number']
    xx = np.arange(len(other_names))
    means = [df_all.query(f'variable == "{name}"')['beta'].mean() for name in other_names]
    sems =  [df_all.query(f'variable == "{name}"')['beta'].sem() * 1.96 for name in other_names]
    ax_others.errorbar(x=xx,
                       y=means, 
                       yerr=sems,
                       ls='none', 
                       marker='o',
                       color='k', 
                       capsize=5, markeredgewidth=1,
                      )
    ax_others.set(xlim=(-0.5, 2.5), ylim=(-0.3, 1))
    
    ax_others.set_xticks(range(len(other_names)))
    ax_others.set_xticklabels(other_names, rotation=45, ha='right')
    ax_others.axhline(y=0, color='k', linestyle=':', linewidth=1)
    ax_others.set(ylabel=r'Linear regression $\beta \pm$95% CI')

    # Back rewards
    xx = np.arange(1, 11)
    means = [df_all.query(f'variable == "reward" and trials_back == {x}')['beta'].mean() for x in xx]
    sems =  [df_all.query(f'variable == "reward" and trials_back == {x}')['beta'].sem() * 1.96 for x in xx]
    
    ax_reward.errorbar(x=xx,
                       y=means, 
                       yerr=sems,
                       ls='-',
                       marker='o',
                       color='k', 
                       capsize=5, 
                       markeredgewidth=1,
                    )
    
    ax_reward.set(xlabel='Reward of past trials')
    ax_reward.axhline(y=0, color='k', linestyle=':', linewidth=1)
    ax_reward.set(xticks=[1, 5, 10], ylim=(-0.35, 0.05))
    ax_reward.invert_yaxis()
    
    sns.despine(trim=True)
    ax.remove()
    
    n_mice = len(df_all['h2o'].unique())
    n_sessions = len(df_all.groupby(['h2o', 'session']).count())
    fig.suptitle(f'Linear regression on RT ({n_mice} mice, {n_sessions} sessions)')

    return fig



if 'df' not in st.session_state: 
    from Home import init
    init()

app()

# try:
#     app()
# except:
#     st.markdown('### Something is wrong. Try going back to 🏠Home or refresh.')