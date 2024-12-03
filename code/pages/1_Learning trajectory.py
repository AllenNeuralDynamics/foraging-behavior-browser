import numpy as np
import pandas as pd
import plotly.graph_objects as go
import s3fs
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from streamlit_plotly_events import plotly_events
from util.aws_s3 import load_data
from util.streamlit import add_session_filter, data_selector

ss = st.session_state

fs = s3fs.S3FileSystem(anon=False)
cache_folder = 'aind-behavior-data/foraging_nwb_bonsai_processed/'


def app():
    
    with st.sidebar:
        add_session_filter(if_bonsai=True)
        data_selector()
        
    if not hasattr(ss, 'df'):
        st.write('##### Data not loaded yet, start from Home:')
        st.page_link('Home.py', label='Home', icon="🏠")
        return
    
    df = load_data()['sessions_bonsai']
    
    # -- get cols --
    col_task = [s for s in df.metadata.columns
                if not any(ss in s for ss in ['lickspout', 'weight', 'water', 'time', 'rig',
                                              'user_name', 'experiment', 'task', 'notes', 'laser']
                )
    ]
    
    col_perf = [s for s in df.session_stats.columns
                if not any(ss in s for ss in ['performance']
                )
    ]
    
    do_pca(ss.df_session_filtered.loc[:, ['subject_id', 'session'] + col_perf], 'performance')
    do_pca(ss.df_session_filtered.loc[:, ['subject_id', 'session'] + col_task], 'task')

    
def do_pca(df, name):
    df = df.dropna(axis=0, how='any')
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
    
    df_to_pca = df.drop(columns=['subject_id', 'session'])
    df_to_pca = df_to_pca.select_dtypes(include=[np.number, float, int])
    
    # Standardize the features
    x = StandardScaler().fit_transform(df_to_pca)
    
    # Apply PCA
    pca = PCA(n_components=10)  # Reduce to 2 dimensions for visualization
    principalComponents = pca.fit_transform(x)
        
    # Create a new DataFrame with the principal components
    principalDf = pd.DataFrame(data=principalComponents)
    principalDf.index = df.set_index(['subject_id', 'session']).index
    
    principalDf.reset_index(inplace=True)
    
    # -- trajectory --
    st.markdown(f'### PCA on {name} metrics')
    fig = go.Figure()

    for mouse_id in principalDf['subject_id'].unique():
        subset = principalDf[principalDf['subject_id'] == mouse_id]
        
        # Add a 3D scatter plot for the current group
        fig.add_trace(go.Scatter3d(
            x=subset[0],
            y=subset[1],
            z=subset[2],
            mode='lines+markers',
            marker=dict(size=subset['session'].apply(
                lambda x: 5 + 15*(x/20))),
            name=f'{mouse_id}',  # Name the trace for the legend
        ))
            
    fig.update_layout(title=name,
                        scene=dict(
                            xaxis_title='Dim1',
                            yaxis_title='Dim2',
                            zaxis_title='Dim3'
                        ),
                        width=1300,
                        height=1000,
                        font_size=15,
                        )
    st.plotly_chart(fig)
    
    # -- variance explained --
    var_explained = pca.explained_variance_ratio_
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(1, len(var_explained)+1),
        y=np.cumsum(var_explained),
        )
    )    
    fig.update_layout(title='Variance Explained',
                      yaxis=dict(range=[0, 1]),
                      width=300,
                      height=400,
                      font_size=15,
                      )
    st.plotly_chart(fig)
    
    # -- pca components --
    pca_components = pd.DataFrame(pca.components_, 
                                  columns=df_to_pca.columns)
    pca_components
    fig = make_subplots(rows=3, cols=1)
    
    # In vertical subplots, each subplot show the components of a principal component
    for i in range(3):
        fig.add_trace(go.Bar(
            x=pca_components.columns,
            y=pca_components.loc[i],
            name=f'PC{i+1}',
        ), row=i+1, col=1)
        
        fig.update_xaxes(showticklabels=i==2, row=i+1, col=1)
        
    fig.update_layout(title='PCA weights',
                      width=1000,
                      height=800,
                      font_size=20,
                      )
    st.plotly_chart(fig)
    
app()