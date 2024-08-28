import base64
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool

from io import BytesIO
from PIL import Image
import numpy as np

import streamlit as st


# https://docs.bokeh.org/en/3.0.2/docs/user_guide/interaction/tools.html#custom-tooltip

import s3fs

# Initialize S3FileSystem
s3 = s3fs.S3FileSystem(anon=False)  # Set anon=True for public buckets

# Specify the S3 path
bucket_name = "aind-behavior-data-dev"
s3_path = f's3://{bucket_name}/foraging_nwb_bonsai_processed/'

# Use glob to find all .png files recursively
png_files = s3.glob(s3_path + '**/*.png')[:100]

# Generate URLs
URLs = [f"https://{bucket_name}.s3.amazonaws.com{file.replace(bucket_name, '')}" for file in png_files]


source = ColumnDataSource(data=dict(
    x=np.random.default_rng().integers(0, 100, len(png_files)),
    y=np.random.default_rng().integers(0, 100, len(png_files)),
    desc=png_files,
    imgs=URLs,
    fonts=[
        '<i>italics</i>',
        '<pre>pre</pre>',
        '<b>bold</b>',
        '<small>small</small>',
        '<del>del</del>'
    ]
))

TOOLTIPS = """
    <div style="width: 600px;">
        <div>
            <img
                src="@imgs" height="200" alt="@imgs" width="600"
                style="display: block; margin: 0 auto 15px auto;"
                border="2"
            ></img>
        </div>
        <div style="text-align: left;">
            <span style="font-size: 17px; font-weight: bold;">@desc</span>
            <span style="font-size: 15px; color: #966;">[$index]</span>
        </div>
        <div style="margin-top: 5px;">
            <span>@fonts{safe}</span>
        </div>
        <div style="margin-top: 5px;">
            <span style="font-size: 15px;">Location</span>
            <span style="font-size: 10px; color: #696;">($x, $y)</span>
        </div>
    </div>
"""

hover = HoverTool(
    tooltips=[("Value", "@y")],
    attachment="right",
    anchor="top_right",
    point_policy="snap_to_data",
)

plot = figure(width=600, height=800,
           tools=[hover, "lasso_select", "reset", "tap", "pan", "wheel_zoom"],
           tooltips=TOOLTIPS,
           title="Mouse over the dots")

plot.circle('x', 'y', size=20, source=source)

# Using the third-party component "streamlit-bokeh3-events"
# https://discuss.streamlit.io/t/bokeh-3-plots-within-streamlit-including-bi-directional-communication/57650
# https://github.com/ChristophNa/stBokeh3Demo/blob/main/pages/1_%F0%9F%A4%A0_Lasso_Selector.py
from streamlit_bokeh3_events import streamlit_bokeh3_events
from bokeh.models import ColumnDataSource, CustomJS

source.selected.js_on_change(
    "indices",
    CustomJS(
        args=dict(source=source),
        code="""
        document.dispatchEvent(
            new CustomEvent("TestSelectEvent", {detail: {indices: cb_obj.indices}})
        )
    """,
    ),
)

event_result = streamlit_bokeh3_events(
    events="TestSelectEvent",
    bokeh_plot=plot,
    key="foo1",
    debounce_time=100,
    refresh_on_update=False,
    override_height=800
)

# some event was thrown
if event_result is not None:
    # TestSelectEvent was thrown
    if "TestSelectEvent" in event_result:
        st.subheader("Selected Points' Pandas Stat summary")
        indices = event_result["TestSelectEvent"].get("indices", [])
        st.table(indices)

st.subheader("Raw Event Data")
st.write(event_result)
