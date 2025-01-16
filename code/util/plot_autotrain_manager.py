from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
pio.json.config.default_engine = "orjson"

import streamlit as st
from aind_auto_train.plot.curriculum import get_stage_color_mapper
from aind_auto_train.schema.curriculum import TrainingStage

from .aws_s3 import get_s3_public_url

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, Rect, CustomJS, LinearAxis, DatetimeTickFormatter
from bokeh.layouts import column

stage_color_mapper = get_stage_color_mapper(stage_list=list(TrainingStage.__members__))

@st.cache_data(ttl=3600 * 12)
def plot_manager_all_progress_bokeh_source(
    x_axis="session",
    sort_by="subject_id",  # "subject_id", "first_date", "last_date", "progress_to_graduated"
    sort_order="descending",  # "ascending", "descending"
    recent_days=None,
    marker_size=10,
    marker_edge_width=2,
    highlight_subjects=[],
    if_show_fig=False,
):
    # --- Prepare data ---
    manager = st.session_state.auto_train_manager
    df_manager = manager.df_manager.sort_values(
        by=["subject_id", "session"],
        ascending=[sort_order == "ascending", False]
    )

    if not len(df_manager):
        return None

    # Metadata merge from df_master
    df_tmp_rig_user_name = st.session_state.df["sessions_bonsai"][
        ["subject_id", "session_date", "session", "rig", "user_name", "nwb_suffix", 
         "foraging_eff_random_seed", "finished_trials", "finished_rate", 
         "task", "curriculum_name", "curriculum_version", "current_stage_actual"]
    ]
    df_tmp_rig_user_name["session_date"] = df_tmp_rig_user_name["session_date"].astype(str)

    df_to_draw = (
        df_manager.drop_duplicates(
            subset=["subject_id", "session_date"], keep="last"
        )  # Duplicte sessions in the autotrain due to pipeline issues
        .drop(
            columns=[
                "session",
                "task",
                "foraging_efficiency", 
                "finished_trials", 
            ]
        )  # df_master has higher priority in session numbers
        .merge(
            df_tmp_rig_user_name.query(f"current_stage_actual != 'None'"),
            on=["subject_id", "session_date"],
            how="right",
        )
    )

    # Correct df_manager missing sessions (df_manager has higher priority in curriculum-related fields)
    df_to_draw["curriculum_name"] = df_to_draw["curriculum_name_x"].fillna(df_to_draw["curriculum_name_y"])
    df_to_draw["curriculum_version"] = df_to_draw["curriculum_version_x"].fillna(df_to_draw["curriculum_version_y"])
    df_to_draw["current_stage_actual"] = df_to_draw["current_stage_actual_x"].fillna(df_to_draw["current_stage_actual_y"])

    df_to_draw["color"] = df_to_draw["current_stage_actual"].map(stage_color_mapper)
    df_to_draw["edge_color"] = (  # Use grey edge to indicate stage without suggestion 
        df_to_draw["current_stage_suggested"].map(stage_color_mapper).fillna("#d3d3d3")
    )
    df_to_draw["imgs_1"] = df_to_draw.apply(
        lambda x: get_s3_public_url(
            subject_id=x["subject_id"],
            session_date=x["session_date"],
            nwb_suffix=x["nwb_suffix"],
            figure_suffix="choice_history.png",
        ),
        axis=1,
    )
    df_to_draw["imgs_2"] = df_to_draw.apply(
        lambda x: get_s3_public_url(
            subject_id=x["subject_id"],
            session_date=x["session_date"],
            nwb_suffix=x["nwb_suffix"],
            figure_suffix="logistic_regression_Su2022.png",
        ),
        axis=1,
    )
    df_to_draw.round(3)

    # --- Filter recent days ---
    df_to_draw['session_date'] = pd.to_datetime(df_to_draw['session_date'])
    if x_axis == 'date' and recent_days is not None:
        date_start = datetime.today() - pd.Timedelta(days=recent_days)
        df_to_draw = df_to_draw.query('session_date >= @date_start')

    # Sort subjects
    if sort_by == "subject_id":
        subject_ids = df_to_draw.subject_id.unique()
    elif sort_by == "first_date":
        subject_ids = df_to_draw.groupby('subject_id').session_date.min().sort_values(ascending=sort_order == "ascending").index
    elif sort_by == "last_date":
        subject_ids = df_to_draw.groupby('subject_id').session_date.max().sort_values(ascending=sort_order == "ascending").index
    elif sort_by == "progress_to_graduated":
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
        raise ValueError("Invalid sort_by value.")

    # Select x
    if x_axis == 'session':
        df_to_draw["x"] = df_to_draw['session']
    elif x_axis == 'date':
        df_to_draw["x"] = df_to_draw['session_date']
    elif x_axis == 'relative_date':
        # groupby subject_id and all subtracted by the min date of each subject
        df_to_draw["x"] = (
            df_to_draw.groupby("subject_id")["session_date"]
            .transform(lambda x: x - x.min())
            .dt.days
        )

    df_to_draw["session_date"] = df_to_draw["session_date"].dt.strftime('%Y-%m-%d')

    # --- Reorder subjects ---
    df_subjects = []

    for n, subject_id in enumerate(subject_ids):
        df_subject = df_to_draw[df_to_draw['subject_id'] == subject_id]
        df_subject["y"] = len(subject_ids) - n
        df_subjects.append(df_subject)

    data_df = pd.concat(df_subjects, ignore_index=True)

    return data_df, subject_ids


def plot_manager_all_progress_bokeh(
    x_axis="session",
    sort_by="subject_id",  # "subject_id", "first_date", "last_date", "progress_to_graduated"
    sort_order="descending",  # "ascending", "descending"
    recent_days=None,
    marker_size=10,
    marker_edge_width=2,
    highlight_subjects=[],
    if_show_fig=False,
):

    data_df, subject_ids = plot_manager_all_progress_bokeh_source(
        x_axis=x_axis,
        sort_by=sort_by,
        sort_order=sort_order,
        recent_days=recent_days,
        marker_size=marker_size,
        marker_edge_width=marker_edge_width,
        highlight_subjects=highlight_subjects,
        if_show_fig=if_show_fig,
    )
    source = ColumnDataSource(data_df)

    # Add hover tool
    TOOLTIPS = """
                <div style="max-width: 1200px; border: 5px solid @color; align-items: top; padding: 10px;">                        
                    <div style="display: flex; flex-direction: row; align-items: top; padding: 10px;">
                        <div style="text-align: left; flex: auto; white-space: nowrap; margin: 0 10px">
                            <span style="font-size: 17px;">
                                <b>Subject: @subject_id</b><br>
                                <b>@session_date, Session @session</b><br>
                                <b>@user_name</b> @ <b>@rig</b><br>
                                <b>@curriculum_name</b><b>_v</b><b>@curriculum_version</b><br>
                                Suggested: <b>@current_stage_suggested</b><br>
                                Actual: <span style="color: @color"><b>@current_stage_actual</b></span><br>
                                <hr style="margin: 5px 0;">
                                Session Task: <b>@task</b><br>
                                Foraging Efficiency: <b>@foraging_eff_random_seed</b><br>
                                Finished Trials: <b>@finished_trials</b><br>
                                Finished Ratio: <b>@finished_rate</b><br>
                                <hr style="margin: 5px 0;">
                                Decision: <b>@decision</b><br>
                                Next Suggested: <b>@next_stage_suggested</b>
                            </span>
                        </div>
                        <div style="text-align: right">
                            <img
                                src="@imgs_2" height="300" alt="@imgs_2" width="350"
                                style="display: block; margin: 10px 10px; border: 1px solid black; border-radius: 5px;">
                        </div>
                    </div>
                    <img
                        src="@imgs_1" height="300" alt="@imgs_1" width="900"
                        style="display: block; margin: 10px 10px; border: 1px solid black; border-radius: 5px;">
                </div>
                """

    # Create Bokeh figure
    p = figure(
        title="AutoTrain Progress",
        x_axis_label=x_axis,
        y_axis_label="Subjects",
        height=20*len(subject_ids),
        width=1400,
        # tools=[hover, "lasso_select", "reset", "tap", "pan", "wheel_zoom"],
        # tooltips=TOOLTIPS,
    )

    scatter_renderer = p.scatter(
        x="x",
        y="y",
        size=marker_size,
        color="color",
        line_color="edge_color",
        line_width=marker_edge_width,
        source=source,
    )

    hover = HoverTool(
        tooltips=TOOLTIPS,
        # attachment="right",
        # anchor="top_right",
        point_policy="snap_to_data",
        # callback=CustomJS(
        #     args=dict(plot=p),
        #     code="""
        #             const tooltip = document.querySelector('.bk-tooltip');
        #             const canvasBounds = plot.frame.canvas_view.el.getBoundingClientRect();
        #             const hoverX = cb_data.geometry.x;  // Hover X position
        #             const tooltipWidth = tooltip.offsetWidth;

        #             if ((hoverX - canvasBounds.left) < tooltipWidth / 2) {
        #                 tooltip.style.left = `${hoverX + 10}px`; // Align tooltip to the right
        #             } else if ((canvasBounds.right - hoverX) < tooltipWidth / 2) {
        #                 tooltip.style.left = `${hoverX - tooltipWidth - 10}px`; // Align tooltip to the left
        #             } else {
        #                 tooltip.style.left = `${hoverX - tooltipWidth / 2}px`; // Center align
        #             }
        #         """,
        # ),
        renderers=[scatter_renderer],
    )

    p.add_tools(
        hover, 
        #"tap" # temporarily disable tap tool
        )

    p.x_range.start = data_df.x.min() - (1 if x_axis != "date" else pd.Timedelta(days=1))
    p.x_range.end = data_df.x.max() + (1 if x_axis != "date" else pd.Timedelta(days=1))
    p.y_range.start = 0
    p.y_range.end = len(subject_ids) + 1

    # Highlight subjects
    y_subjec_id_mapper = {len(subject_ids) - i: subject_ids[i] for i in range(len(subject_ids))}

    for subject_id in highlight_subjects:
        y = {v: k for k, v in y_subjec_id_mapper.items()}.get(subject_id)
        if y is None: continue
        rect = Rect(
            x=(
                (p.x_range.end + p.x_range.start) / 2
                if x_axis != "date"
                else pd.Timestamp((p.x_range.end.value + p.x_range.start.value) // 2)
            ),
            width=p.x_range.end - p.x_range.start,
            y=y,
            height=1,
            fill_color="blue",
            fill_alpha=0.1,
            line_width=0,
        )
        p.add_glyph(rect)

    # Customize the plot
    p.yaxis.ticker = np.arange(1, len(subject_ids)+1)  # Tick positions corresponding to y values
    p.yaxis.major_label_overrides = y_subjec_id_mapper  # Map numeric ticks to string labels

    top_axis = LinearAxis(axis_label=x_axis)
    p.add_layout(top_axis, "above")  # Add the new axis to the "above" location
    if x_axis == "date":
        p.xaxis.formatter = DatetimeTickFormatter(
            days="%b %d, %Y",  # Format for days
            months="%b %Y",    # Format for months
            years="%Y",        # Format for years
        )

    # Font sizes
    p.title.text_font_size = "16pt"  # Title font size
    p.xaxis.axis_label_text_font_size = "14pt"  # X-axis label font size
    p.yaxis.axis_label_text_font_size = "14pt"  # Y-axis label font size
    p.xaxis.major_label_text_font_size = "12pt"  # X-axis tick font size
    p.yaxis.major_label_text_font_size = "12pt"  # Y-axis tick font size

    # Add callback for selection
    # source.selected.js_on_change(
    #     "indices",
    #     CustomJS(
    #         args=dict(source=source),
    #         code="""
    #         document.dispatchEvent(
    #             new CustomEvent("TestSelectEvent", {detail: {indices: cb_obj.indices}})
    #         )
    #     """,
    #     ),
    # )

    if if_show_fig:
        show(p)

    return p, data_df


@st.cache_data(ttl=3600 * 24)
def plot_manager_all_progress(
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

    manager = st.session_state.auto_train_manager

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
