from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from aind_auto_train.plot.curriculum import get_stage_color_mapper
from aind_auto_train.schema.curriculum import TrainingStage

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, ColorBar, LinearColorMapper
from bokeh.layouts import gridplot

def plot_manager_all_progress_bokeh(
    manager: "AutoTrainManager",
    x_axis: ["session", "date", "relative_date"] = "session",  # type: ignore
    sort_by: [
        "subject_id",
        "first_date",
        "last_date",
        "progress_to_graduated",
    ] = "subject_id",
    sort_order: ["ascending", "descending"] = "descending",
    recent_days: int = None,
    marker_size=10,
    marker_edge_width=2,
    highlight_subjects=[],
    if_show_fig=True,
):

    # Set default order
    df_manager = manager.df_manager.sort_values(by=['subject_id', 'session'],
                                                ascending=[sort_order == 'ascending', False])

    if not len(df_manager):
        return None

    # Get some additional metadata from the master table
    df_tmp_rig_user_name = st.session_state.df['sessions_bonsai'].loc[:, ['subject_id', 'session_date', 'rig', 'user_name']]
    df_tmp_rig_user_name.session_date = df_tmp_rig_user_name.session_date.astype(str)

    # Sort mice
    if sort_by == 'subject_id':
        subject_ids = df_manager.subject_id.unique()
    elif sort_by == 'first_date':
        subject_ids = pd.DataFrame(df_manager.groupby('subject_id').session_date.min()
                                   ).reset_index().sort_values(
                                       by=['session_date', 'subject_id'],
                                       ascending=sort_order == 'ascending').subject_id
    elif sort_by == 'last_date':
        subject_ids = pd.DataFrame(df_manager.groupby('subject_id').session_date.max()
                                    ).reset_index().sort_values(
                                            by=['session_date', 'subject_id'],
                                            ascending=sort_order == 'ascending').subject_id
    elif sort_by == 'progress_to_graduated':
        manager.compute_stats()
        df_stats = manager.df_manager_stats

        # Sort by 'first_entry' of GRADUATED
        subject_ids = df_stats.reset_index().set_index(
            'subject_id'
        ).query(
            f'current_stage_actual == "GRADUATED"'
        )['first_entry'].sort_values(
            ascending=sort_order != 'ascending').index.to_list()

        # Append subjects that have not graduated
        subject_ids = subject_ids + [s for s in df_manager.subject_id.unique() if s not in subject_ids]

    else:
        raise ValueError(
            f'sort_by must be in {["subject_id", "first_date", "last_date", "progress"]}')

    # Preparing the scatter plot
    plots = []
    for n, subject_id in enumerate(subject_ids):
        df_subject = df_manager[df_manager['subject_id'] == subject_id]

        # Get stage_color_mapper
        stage_color_mapper = get_stage_color_mapper(stage_list=list(TrainingStage.__members__))

        # Get h2o if available
        if 'h2o' in manager.df_behavior:
            h2o = manager.df_behavior[
                manager.df_behavior['subject_id'] == subject_id]['h2o'].iloc[0]
        else:
            h2o = None

        df_subject = df_subject.merge(
            df_tmp_rig_user_name,
            on=['subject_id', 'session_date'], how='left')

        # Handle open loop sessions
        open_loop_ids = df_subject.if_closed_loop == False
        color_actual = df_subject['current_stage_actual'].map(
            stage_color_mapper)
        color_actual[open_loop_ids] = 'lightgrey'
        stage_actual = df_subject.current_stage_actual.values
        stage_actual[open_loop_ids] = 'unknown (open loop)'

        # Select x
        if x_axis == 'session':
            x = df_subject['session']
        elif x_axis == 'date':
            x = pd.to_datetime(df_subject['session_date'])
        elif x_axis == 'relative_date':
            x = pd.to_datetime(df_subject['session_date'])
            x = (x - x.min()).dt.days
        else:
            raise ValueError(
                f"x_axis can only be in ['session', 'date', 'relative_date']")

        # Cache x range
        xrange_min = x.min() if n == 0 else min(x.min(), xrange_min)
        xrange_max = x.max() if n == 0 else max(x.max(), xrange_max)

        y = len(subject_ids) - n  # Y axis

        source = ColumnDataSource(data=dict(
            x=x,
            y=[y] * len(df_subject),
            stage_actual=stage_actual,
            curriculum_name=df_subject.curriculum_name,
            curriculum_version=df_subject.curriculum_version,
            rig=df_subject.rig,
            user_name=df_subject.user_name,
            session=df_subject.session,
            session_date=df_subject.session_date,
            foraging_efficiency=df_subject.foraging_efficiency,
            finished_trials=df_subject.finished_trials,
            task=df_subject.task
        ))

        p = figure(plot_width=900, plot_height=200, title=f"Mouse {subject_id}",
                   x_axis_label=x_axis, y_axis_label='Mouse', tools='pan,wheel_zoom,box_zoom,reset,save')

        p.scatter(x='x', y='y', size=marker_size, source=source,
                  fill_alpha=0.6, line_width=marker_edge_width, color=color_actual)

        hover = HoverTool()
        hover.tooltips = [
            ("Subject", "@stage_actual"),
            ("Curriculum Name", "@curriculum_name"),
            ("Curriculum Version", "@curriculum_version"),
            ("Rig", "@rig"),
            ("User Name", "@user_name"),
            ("Session", "@session"),
            ("Session Date", "@session_date"),
            ("Foraging Efficiency", "@foraging_efficiency"),
            ("Finished Trials", "@finished_trials"),
            ("Task", "@task")
        ]
        p.add_tools(hover)

        plots.append(p)

    # Create grid plot
    grid = gridplot(plots, ncols=1)

    # Show the plot
    if if_show_fig:
        show(grid)

    return grid


def plot_manager_all_progress(manager: 'AutoTrainManager',
                              x_axis: ['session', 'date',
                                       'relative_date'] = 'session', # type: ignore
                              sort_by: ['subject_id', 'first_date',
                                        'last_date', 'progress_to_graduated'] = 'subject_id',
                              sort_order: ['ascending',
                                           'descending'] = 'descending',
                              recent_days: int=None,
                              marker_size=10,
                              marker_edge_width=2,
                              highlight_subjects=[],
                              if_show_fig=True
                              ):
    
    
    # %%
    # Set default order
    df_manager = manager.df_manager.sort_values(by=['subject_id', 'session'],
                                                ascending=[sort_order == 'ascending', False])
    
    if not len(df_manager):
        return None
    
    # Get some additional metadata from the master table
    df_tmp_rig_user_name = st.session_state.df['sessions_bonsai'].loc[:, ['subject_id', 'session_date', 'rig', 'user_name']]
    df_tmp_rig_user_name.session_date = df_tmp_rig_user_name.session_date.astype(str)

    # Sort mice
    if sort_by == 'subject_id':
        subject_ids = df_manager.subject_id.unique()
    elif sort_by == 'first_date':
        subject_ids = pd.DataFrame(df_manager.groupby('subject_id').session_date.min()
                                   ).reset_index().sort_values(
                                       by=['session_date', 'subject_id'],
                                       ascending=sort_order == 'ascending').subject_id
    elif sort_by == 'last_date':
        subject_ids = pd.DataFrame(df_manager.groupby('subject_id').session_date.max()
                                    ).reset_index().sort_values(
                                            by=['session_date', 'subject_id'],
                                            ascending=sort_order == 'ascending').subject_id
    elif sort_by == 'progress_to_graduated':
        manager.compute_stats()
        df_stats = manager.df_manager_stats
        
        # Sort by 'first_entry' of GRADUATED
        subject_ids = df_stats.reset_index().set_index(
            'subject_id'
        ).query(
            f'current_stage_actual == "GRADUATED"'
        )['first_entry'].sort_values(
            ascending=sort_order != 'ascending').index.to_list()
        
        # Append subjects that have not graduated
        subject_ids = subject_ids + [s for s in df_manager.subject_id.unique() if s not in subject_ids]
        
    else:
        raise ValueError(
            f'sort_by must be in {["subject_id", "first_date", "last_date", "progress"]}')

    # Preparing the scatter plot
    traces = []
    for n, subject_id in enumerate(subject_ids):
        df_subject = df_manager[df_manager['subject_id'] == subject_id]
        
        # Get stage_color_mapper
        stage_color_mapper = get_stage_color_mapper(stage_list=list(TrainingStage.__members__))
        
        # Get h2o if available
        if 'h2o' in manager.df_behavior:
            h2o = manager.df_behavior[
                manager.df_behavior['subject_id'] == subject_id]['h2o'].iloc[0]
        else:
            h2o = None
            
        df_subject = df_subject.merge(
            df_tmp_rig_user_name,
            on=['subject_id', 'session_date'], how='left')

        # Handle open loop sessions
        open_loop_ids = df_subject.if_closed_loop == False
        color_actual = df_subject['current_stage_actual'].map(
            stage_color_mapper)
        color_actual[open_loop_ids] = 'lightgrey'
        stage_actual = df_subject.current_stage_actual.values
        stage_actual[open_loop_ids] = 'unknown (open loop)'

        # Select x
        if x_axis == 'session':
            x = df_subject['session']
        elif x_axis == 'date':
            x = pd.to_datetime(df_subject['session_date'])
        elif x_axis == 'relative_date':
            x = pd.to_datetime(df_subject['session_date'])
            x = (x - x.min()).dt.days
        else:
            raise ValueError(
                f"x_axis can only be in ['session', 'date', 'relative_date']")
        
        # Cache x range
        xrange_min = x.min() if n == 0 else min(x.min(), xrange_min)
        xrange_max = x.max() if n == 0 else max(x.max(), xrange_max)
        
        y = len(subject_ids) - n  # Y axis

        traces.append(go.Scattergl(
            x=x,
            y=[y] * len(df_subject),
            mode='markers',
            marker=dict(
                size=marker_size,
                line=dict(
                    width=marker_edge_width,
                    color=df_subject['current_stage_suggested'].map(
                        stage_color_mapper)
                ),
                color=color_actual,
                # colorbar=dict(title='Training Stage'),
            ),
            name=f'Mouse {subject_id}',
            hovertemplate=(f"<b>Subject {subject_id} ({h2o})</b>"
                           "<br><b>Session %{customdata[0]}, %{customdata[1]}</b>"
                           "<br><b>%{customdata[12]} @ %{customdata[11]}</b>"
                           "<br><b>%{customdata[2]}_v%{customdata[3]}</b>"
                           "<br>Suggested: <b>%{customdata[4]}</b>"
                           "<br>Actual: <b>%{customdata[5]}</b>"
                           f"<br>{'-'*10}"
                           "<br>Session task: <b>%{customdata[6]}</b>"
                           "<br>foraging_eff = %{customdata[7]}"
                           "<br>finished_trials = %{customdata[8]}"
                           f"<br>{'-'*10}"
                           "<br>Decision = <b>%{customdata[9]}</b>"
                           "<br>Next suggested: <b>%{customdata[10]}</b>"
                           "<extra></extra>"),
            customdata=np.stack(
                (df_subject.session,
                 df_subject.session_date,
                 df_subject.curriculum_name,
                 df_subject.curriculum_version,
                 df_subject.current_stage_suggested,
                 stage_actual, # 5
                 df_subject.task,
                 np.round(df_subject.foraging_efficiency, 3),
                 df_subject.finished_trials,
                 df_subject.decision,
                 df_subject.next_stage_suggested, # 10
                 df_subject.rig, # 11
                 df_subject.user_name, # 12
                 ), axis=-1),
            showlegend=False
        )
        )
        
        # Add "x" for open loop sessions
        traces.append(go.Scattergl(
            x=x[open_loop_ids],
            y=[y] * len(df_subject),
            mode='markers',
            marker=dict(
                size=marker_size*0.8,
                symbol='x-thin',
                color='black',
                line_width=marker_edge_width*0.8,
            ),
            showlegend=False,
        )
        )

    # Create the figure
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Automatic training progress ({manager.manager_name})",
        xaxis_title=x_axis,
        yaxis_title='Mouse',
        height=1200,
    )

    # Set subject_id as y axis label
    fig.update_layout(
        hovermode='closest',
        yaxis=dict(
            tickmode='array',
            tickvals=np.arange(len(subject_ids), 0, -1), 
            ticktext=subject_ids, 
            # autorange='reversed',  # This will lead to a weird space at the top
            zeroline=False,
            title=''
        ),
        yaxis_range=[-0.5, len(subject_ids) + 1],
    )
    
    # Limit x range to recent days if x is "date"
    if x_axis == 'date' and recent_days is not None:
        # xrange_max = pd.Timestamp.today()  # For unknown reasons, using this line will break both plotly_events and new st.plotly_chart callback...
        xrange_max = pd.to_datetime(df_manager.session_date).max() + pd.Timedelta(days=1)  
        xrange_min = xrange_max - pd.Timedelta(days=recent_days)        
        fig.update_layout(xaxis_range=[xrange_min, xrange_max])
    
    # Highight the selected subject
    for n, subject_id in enumerate(subject_ids):
        y = len(subject_ids) - n  # Y axis
        if subject_id in highlight_subjects:
            fig.add_shape(
                type="rect",
                y0=y-0.5,  
                y1=y+0.5,
                x0=xrange_min - (1 if x_axis != 'date' else pd.Timedelta(days=1)),
                x1=xrange_max + (1 if x_axis != 'date' else pd.Timedelta(days=1)),
                line=dict(
                    width=0,
                ),
                fillcolor="Gray",
                opacity=0.3,
                layer="below"
            )
    

    # Show the plot
    if if_show_fig:
        fig.show()

    # %%
    return fig
