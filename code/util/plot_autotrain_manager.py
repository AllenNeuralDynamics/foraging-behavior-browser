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
    if_use_filtered_data=True,  # Use data filtered from the sidebar
    filtered_session_ids=None,
):
    # --- Prepare data ---
    # Now we already merged full curriculum info in the master df_session by the result access api
    # manager = st.session_state.auto_train_manager

    df_to_draw = st.session_state.df['sessions_main'].sort_values(
        by=["subject_id", "session"],
        ascending=[sort_order == "ascending", False]
    ).copy()

    if not len(df_to_draw):
        return None

    # If use_filtered_data, filter the data
    if if_use_filtered_data:
        df_to_draw = df_to_draw.merge(
            filtered_session_ids,
            on=["subject_id", "session"],
            how="inner",
        )

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
    
    # --- Remove rows with NaN in color or edge_color ---
    # to fix a bug where non-normalized stages appears in the autotrain table
    df_to_draw = df_to_draw.dropna(subset=["color", "edge_color"])

    # --- Filter recent days ---
    df_to_draw['session_date'] = pd.to_datetime(df_to_draw['session_date'])
    if not if_use_filtered_data and x_axis == 'date' and recent_days is not None:
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
        # Only subjects in df_to_draw
        subject_ids = [s for s in subject_ids if s in df_to_draw.subject_id.unique()]
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

    if not df_subjects:
        return None, None
    
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
    if_use_filtered_data=True,  # Use data filtered from the sidebar
    filtered_session_ids=None,
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
        if_use_filtered_data=if_use_filtered_data,
        filtered_session_ids=filtered_session_ids,
    )
    
    if data_df is None:
        return None, None
    
    source = ColumnDataSource(data_df)

    # Add hover tool
    TOOLTIPS = """
                <div style="max-width: 1200px; border: 5px solid @color; align-items: top; padding: 10px;">                        
                    <div style="display: flex; flex-direction: row; align-items: top; padding: 10px;">
                        <div style="text-align: left; flex: auto; white-space: nowrap; margin: 0 10px">
                            <span style="font-size: 17px;">
                                <b>Subject: @subject_id (@PI)</b><br>
                                <b>@session_date, Session @session</b><br>
                                <b>@trainer</b> @ <b>@rig</b><br>
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
        title="AutoTrain Progress"
        + (
            " (all on-curriculum sessions)"
            if not if_use_filtered_data
            else " (sessions filtered on the side bar)"
        ),
        x_axis_label=x_axis,
        y_axis_label="Subjects",
        height=max(770, 30 * len(subject_ids)),
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
