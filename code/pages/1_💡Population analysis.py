import streamlit as st

import importlib
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

from streamlit_util import filter_dataframe, aggrid_interactive_table_session, add_session_filter
from population_util import _draw_variable_trial_back



def app():

    with st.sidebar:
        add_session_filter()
        
        
    # st.dataframe(st.session_state.df['logistic_regression'])
    
    
    df_to_population = st.session_state.df_session_filtered

    with st.columns([1, 7])[0]:
        beta_names = st.multiselect('beta names', ['RewC', 'UnrC', 'C'], ['RewC', 'UnrC', 'C'])
        max_trials_back = st.slider('max trials back', 1, 5, 3)

    if st.button('Plot photostim logistic regression:'):
        df_all = df_to_population.merge(st.session_state.df['logistic_regression'], on=('subject_id', 'session'), how='inner')
        fig = plot_logistic_regression_photostim(df_all, beta_names=beta_names, past_trials_to_plot=range(1, max_trials_back + 1))
        
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf, width=3000)
        
        
    st.dataframe(st.session_state.df['linear_regression_rt'])

    
    
    # st.pyplot(fig)
    

@st.cache_data(ttl=3600*24)
def plot_logistic_regression_photostim(df_all, beta_names=['RewC', 'UnrC', 'C'], past_trials_to_plot=(1, 2, 3)):
    
    df_all['error_bar'] = [[x, y] for x, y in zip(df_all['mean'] - df_all.lower_ci,  df_all.upper_ci - df_all['mean'])]
    df_all['h2o_session'] = df_all[['h2o', 'session']].astype(str).apply('_'.join, axis=1)
    
    df_all.query('trial_group != "all_no_stim"', inplace=True)


    fig, axes = plt.subplots(len(past_trials_to_plot), len(beta_names) + 1, 
                            figsize=(10 * (len(beta_names) + 1), 5 * len(past_trials_to_plot)), constrained_layout=False,
                            dpi=70,
                            gridspec_kw=dict(hspace=0.3, wspace=0.3, top=0.9, bottom=0.1, left=0.1, right=0.9),
                           )
    axes = np.atleast_2d(axes)

    for i, trials_back in enumerate(past_trials_to_plot):
        for j, name in enumerate(beta_names):        
            _draw_variable_trial_back(df_all, name, trials_back, ax=axes[i, j])
    _draw_variable_trial_back(df_all, 'bias', 0, ax=axes[0, j + 1])

    for i in range(1, len(past_trials_to_plot)): axes[i, -1].remove()
    
    return fig


if 'df' not in st.session_state: 
    from Home import init
    init()

app()

# try:
#     app()
# except:
#     st.markdown('### Something is wrong. Try going back to 🏠Home or refresh.')