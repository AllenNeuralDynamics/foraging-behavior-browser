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
    
    
    df_to_population = st.session_state.df_session_filtered

    with st.columns([1, 7])[0]:
        beta_names = st.multiselect('beta names', ['RewC', 'UnrC', 'C'], ['RewC', 'UnrC', 'C'])
        max_trials_back = st.slider('max trials back', 1, 5, 3)

    if st.button('Plot photostim logistic regression'):
        df_all = df_to_population.merge(st.session_state.df['logistic_regression'], on=('subject_id', 'session'), how='inner')
        fig = plot_logistic_regression_photostim(df_all, beta_names=beta_names, past_trials_to_plot=range(1, max_trials_back + 1))
        
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf, width=3000)
    
    if st.button('Plot photostim linear regression on RT'):
        df_all = df_to_population.merge(st.session_state.df['linear_regression_rt'], on=('subject_id', 'session'), how='inner')
        df_all.query('trial_group != "all_no_stim"', inplace=True)
        # df_all
        
        fig = plot_linear_regression_rt_photostim(df_all, beta_names=['reward_1', 'reward_2', 'previous_iti', 'trial_number', 'this_choice', 'constant'])
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)
        
    if st.button('Plot logistic regression for non-stim sessions'):
        df_all = df_to_population.merge(st.session_state.df['linear_regression_rt'], on=('subject_id', 'session'), how='inner')
        df_all.query('trial_group == "all_no_stim"', inplace=True)
        df_all
        
        fig = plot_linear_regression_rt_non_photostim(df_all)
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf, width=1000)
    
    # st.pyplot(fig)
    

@st.cache_data(ttl=3600*24)
def plot_logistic_regression_photostim(df_all, beta_names=['RewC', 'UnrC', 'C'], past_trials_to_plot=(1, 2, 3)):
    
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


# @st.cache_data(ttl=3600*24)
def plot_linear_regression_rt_photostim(df_all, beta_names):
        
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
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=200,
                               gridspec_kw=dict(bottom=0.3))
        
    gs = ax._subplotspec.subgridspec(1, 2, width_ratios=[1, 3], wspace=0.3)
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
    ax_others.set_xlim(-0.5, 2.5)
    
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
    ax_reward.set(xticks=[1, 5, 10])
    ax_reward.invert_yaxis()
    
    sns.despine(trim=True)
    ax.remove()
    
    n_mice = len(df_all['h2o'].unique())
    n_sessions = len(df_all.groupby(['h2o', 'session']).count())
    fig.suptitle(f'Linear regression on RT, {n_mice} mice, {n_sessions} sessions')

    
    return fig



if 'df' not in st.session_state: 
    from Home import init
    init()

app()

# try:
#     app()
# except:
#     st.markdown('### Something is wrong. Try going back to 🏠Home or refresh.')