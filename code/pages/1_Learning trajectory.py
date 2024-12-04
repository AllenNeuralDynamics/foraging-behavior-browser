import numpy as np
import pandas as pd
import plotly.graph_objects as go
import s3fs
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from streamlit_plotly_events import plotly_events
from util.aws_s3 import load_data
from util.streamlit import add_session_filter, data_selector
from scipy.stats import gaussian_kde
import streamlit_nested_layout

import extra_streamlit_components as stx

from util.url_query_helper import (
    checkbox_wrapper_for_url_query,
    multiselect_wrapper_for_url_query,
    number_input_wrapper_for_url_query,
    slider_wrapper_for_url_query,
    sync_session_state_to_URL,
    sync_URL_to_session_state,
)


from Home import init

ss = st.session_state

fs = s3fs.S3FileSystem(anon=False)
cache_folder = "aind-behavior-data/foraging_nwb_bonsai_processed/"

try:
    st.set_page_config(
        layout="wide",
        page_title="Foraging behavior browser",
        page_icon=":mouse2:",
        menu_items={
            "Report a bug": "https://github.com/AllenNeuralDynamics/foraging-behavior-browser/issues",
            "About": "Github repo: https://github.com/AllenNeuralDynamics/foraging-behavior-browser",
        },
    )
except:
    pass

# Sort stages in the desired order
STAGE_ORDER = [
    "STAGE_1_WARMUP",
    "STAGE_1",
    "STAGE_2",
    "STAGE_3",
    "STAGE_4",
    "STAGE_FINAL",
    "GRADUATED",
]

@st.cache_data()
def get_stage_color_mapper(stage_list):
    # Mapping stages to colors from red to green, return rgb values
    # Interpolate between red and green using the number of stages
    cmap = plt.cm.get_cmap('RdYlGn', 100)
    stage_color_mapper = {
        stage: matplotlib.colors.rgb2hex(
            cmap(i / (len(stage_list) - 1))) 
        for i, stage in enumerate(stage_list)
    }
    return stage_color_mapper

STAGE_COLOR_MAPPER = get_stage_color_mapper(STAGE_ORDER)

@st.cache_data()
def _get_metadata_col():
    df = load_data()["sessions_bonsai"]

    # -- get cols --
    col_task = [
        s
        for s in df.metadata.columns
        if not any(
            ss in s
            for ss in [
                "lickspout",
                "weight",
                "water",
                "time",
                "rig",
                "user_name",
                "experiment",
                "task",
                "notes",
                "laser",
                "commit",
                "repo",
                "branch",
            ]  # exclude some columns
        )
    ]

    col_perf = [
        s
        for s in df.session_stats.columns
        if not any(ss in s for ss in ["performance"])
    ]
    return col_perf, col_task

COL_PERF, COL_TASK = _get_metadata_col()

def app():
    with st.sidebar:
        add_session_filter(if_bonsai=True)
        data_selector()

    if not hasattr(ss, "df"):
        st.write("##### Data not loaded yet, start from Home:")
        st.page_link("Home.py", label="Home", icon="🏠")
        return

    # === Main tabs ===
    chosen_id = stx.tab_bar(
        data=[
            stx.TabBarItemData(
                id="tab_PCA",
                title="PCA",
                description="PCA on performance and task parameters",
            ),
            stx.TabBarItemData(
                id="tab_stage",
                title="Training stages",
                description="Compare across training stages",
            ),
        ],
        default=(
            st.query_params["tab_id_learning_trajectory"]
            if "tab_id_learning_trajectory" in st.query_params
            else st.session_state.tab_id_learning_trajectory
        ),
    )

    placeholder = st.container()
    st.session_state.tab_id_learning_trajectory = chosen_id

    if chosen_id == "tab_PCA":
        do_pca(
            ss.df_session_filtered.loc[:, ["subject_id", "session"] + COL_PERF],
            "performance",
        )
        do_pca(
            ss.df_session_filtered.loc[:, ["subject_id", "session"] + COL_TASK], "task"
        )
    elif chosen_id == "tab_stage":
        st.markdown("### Distributions of metrics and/or parameters grouped by training stages")
        metrics_grouped_by_stages(df=ss.df_session_filtered)

    # Update back to URL
    sync_session_state_to_URL()


def do_pca(df, name):
    df = df.dropna(axis=0, how="any")
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]

    df_to_pca = df.drop(columns=["subject_id", "session"])
    df_to_pca = df_to_pca.select_dtypes(include=[np.number, float, int])

    # Standardize the features
    x = StandardScaler().fit_transform(df_to_pca)

    # Apply PCA
    pca = PCA(n_components=10)  # Reduce to 2 dimensions for visualization
    principalComponents = pca.fit_transform(x)

    # Create a new DataFrame with the principal components
    principalDf = pd.DataFrame(data=principalComponents)
    principalDf.index = df.set_index(["subject_id", "session"]).index

    principalDf.reset_index(inplace=True)

    # -- trajectory --
    st.markdown(f"### PCA on {name} metrics")
    fig = go.Figure()

    for mouse_id in principalDf["subject_id"].unique():
        subset = principalDf[principalDf["subject_id"] == mouse_id]

        # Add a 3D scatter plot for the current group
        fig.add_trace(
            go.Scatter3d(
                x=subset[0],
                y=subset[1],
                z=subset[2],
                mode="lines+markers",
                marker=dict(size=subset["session"].apply(lambda x: 5 + 15 * (x / 20))),
                name=f"{mouse_id}",  # Name the trace for the legend
            )
        )

    fig.update_layout(
        title=name,
        scene=dict(xaxis_title="Dim1", yaxis_title="Dim2", zaxis_title="Dim3"),
        width=1300,
        height=1000,
        font_size=15,
    )
    st.plotly_chart(fig)

    # -- variance explained --
    var_explained = pca.explained_variance_ratio_
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, len(var_explained) + 1),
            y=np.cumsum(var_explained),
        )
    )
    fig.update_layout(
        title="Variance Explained",
        yaxis=dict(range=[0, 1]),
        width=300,
        height=400,
        font_size=15,
    )
    st.plotly_chart(fig)

    # -- pca components --
    pca_components = pd.DataFrame(pca.components_, columns=df_to_pca.columns)
    pca_components
    fig = make_subplots(rows=3, cols=1)

    # In vertical subplots, each subplot show the components of a principal component
    for i in range(3):
        fig.add_trace(
            go.Bar(
                x=pca_components.columns,
                y=pca_components.loc[i],
                name=f"PC{i+1}",
            ),
            row=i + 1,
            col=1,
        )

        fig.update_xaxes(showticklabels=i == 2, row=i + 1, col=1)

    fig.update_layout(
        title="PCA weights",
        width=1000,
        height=800,
        font_size=20,
    )
    st.plotly_chart(fig)


def metrics_grouped_by_stages(df):
                
    df["current_stage_actual"] = pd.Categorical(
        df["current_stage_actual"], categories=STAGE_ORDER, ordered=True
    )
    df = df.sort_values("current_stage_actual")

    # Multiselect for choosing numeric columns
    cols = st.columns([1, 1, 1])
    
    selected_perf_columns = multiselect_wrapper_for_url_query(
        st,
        label= "Animal performance to plot",
        options=COL_PERF,
        default=["finished_trials", "finished_rage", "foraging_eff_random_seed"],
        key='stage_distribution_selected_perf_columns',
    )
    selected_task_columns = multiselect_wrapper_for_url_query(
        st,
        label= "Task parameters to plot",
        options=COL_TASK,
        default=["effective_block_length_median", "duration_iti_mean", "p_reward_contrast_mean"],
        key='stage_distribution_selected_task_columns',
    )            
    selected_columns = selected_perf_columns + selected_task_columns

    # Checkbox to use density or not
    use_kernel_smooth = st.checkbox("Use Kernel Smoothing", value=True)
    if use_kernel_smooth:
        use_density = False
        bins = 100
    else:
        bins = st.columns([1, 5])[0].slider("Number of bins", 10, 100, 20, 5)
        use_density = st.checkbox("Use Density", value=False)

    # Create a density plot for each selected column grouped by 'current_stage_actual'
    for column in selected_columns:
        fig = _plot_histograms(df, column, bins, use_kernel_smooth, use_density)
        st.plotly_chart(fig)
        
@st.cache_data()
def _plot_histograms(df, column, bins, use_kernel_smooth, use_density):
    fig = go.Figure()
    
    stage_data_all = df[column].dropna()
    stage_data_all = stage_data_all[~stage_data_all.isin([np.inf, -np.inf])]
    bin_edges = np.linspace(stage_data_all.min(), stage_data_all.max(), bins)            
    
    for stage in df["current_stage_actual"].cat.categories:
        if stage not in df["current_stage_actual"].unique():
            continue
        stage_data = df[df["current_stage_actual"] == stage][column].dropna()
        count = len(stage_data)
        if use_kernel_smooth:
            kde = gaussian_kde(stage_data)
            y_vals = kde(bin_edges)
        else:
            y_vals, _ = np.histogram(stage_data, bins=bin_edges, density=use_density)
        percentiles = [(np.sum(stage_data <= x) / len(stage_data)) * 100 for x in bin_edges[1:]]
        customdata = np.array([percentiles]).T
        
        fig.add_trace(
            go.Scatter(
                x=(bin_edges[1:] + bin_edges[:-1]) / 2, 
                y=y_vals, 
                mode="lines",
                line=dict(color=STAGE_COLOR_MAPPER[stage]),
                name=f"{stage} (n={count})",
                customdata=customdata,
                hovertemplate=f"Percentile: %{{customdata[0]:.2f}}%<br><extra></extra>"
            )
        )
    fig.update_layout(
        title=f'{"Animal performance: " if column in COL_PERF else "Task parameters: "}{column}',
        xaxis_title=column,
        yaxis_title="Kernel density" if use_kernel_smooth else "Density" if use_density else "Count",
        hovermode="x unified",
    )
    return fig

if "df" not in st.session_state or "sessions_bonsai" not in st.session_state.df.keys():
    init(if_load_docDB=False)

app()
