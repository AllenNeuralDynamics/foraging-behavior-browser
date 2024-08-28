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
    <div>
        <div>
            <img
                src="@imgs" height="200" alt="@imgs" width="600"
                style="float: right; margin: 0px 15px 15px 0px;"
                border="2"
            ></img>
        </div>
        <div>
            <span style="font-size: 17px; font-weight: bold;">@desc</span>
            <span style="font-size: 15px; color: #966;">[$index]</span>
        </div>
        <div>
            <span>@fonts{safe}</span>
        </div>
        <div>
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

p = figure(width=600, height=700,
           tools=[hover],
           tooltips=TOOLTIPS,
           title="Mouse over the dots")

p.circle('x', 'y', size=20, source=source)

# Display the plot
st.bokeh_chart(p)
