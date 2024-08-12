import base64
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool
from io import BytesIO
from PIL import Image

import streamlit as st

source = ColumnDataSource(
        data=dict(
            x=[1, 2, 3, 4, 5],
            y=[2, 5, 8, 2, 7],
            desc=['A', 'b', 'C', 'd', 'E'],
            imgs = [
                'https://aind-behavior-data.s3.us-west-2.amazonaws.com/foraging_nwb_bonsai_processed/662914_2023-10-05/662914_2023-10-05_choice_history.png?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEJr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLXdlc3QtMiJGMEQCIBa5%2BAP44jxOBSMrwe5cDt4QnIyV4P1Tl2F5uCtUFMZKAiAvH0sBvBHupG4neCFmLbNIq1l12wwXTtfAB46k%2FoBkeiqEAwiT%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAAaDDQ2NzkxNDM3ODAwMCIMipNV2Ja%2B1mZpOlXDKtgCx%2BmZExJfzyeL7j6q%2BY2x4g9YE7Pd%2FjWX2B2GJHGj5ueHKJxDulAoYz52CFpIou7FHufEqdxs0%2FuvcNoBam8mlcLOcvVhXXgTvdfCpk%2BySQiFhqJyxmscwKePPkzV39n2h6FJfj6D0ZvQ1YTG%2Fg8kVbNR5quEnOU7jVdUCOsbA4X9keGDM%2BE4DXIJS8TF5jEUcxfT6tDPLdaF4U81Jqc86Mcq3QC0EFXO0xvzAraXhkpujKyoWAhhzH4c8vZiQGVIZJrFD%2FR9ek%2BHA%2FcXoWb%2B9YxPM8SaSTc0vn%2B33mvi3L1x%2FfzsQ6k9GhzMOpZn5aNPZI9w4707HaILG7p4gN1kP4bqOLnRn0dooWQd5lmqQ%2BHnzTjIMWBKvugDIIJJbQrVz9WfF1nUHQVomG6zAelKOxbqJZi5GOHf15nN7rBL8NSpGvx53NGhGz1TwtHsybO%2FUxkjRTwQoukw15nptQY6tAKo0y3vh%2FDCh85ecP1xdwEACnMoV9aiA6wNw2%2BrDw%2BsyVKvTFR3lQ%2B6i8y1ElS6BmUb7S2ztOVyzETp2d8i%2FRgM7FC%2FKtlnwYZkFwse2jtNDHA73rONA7wWYhKX9JVVYmaTpqcoPeFAFUFpBkzT16D3OCfU8J1nN%2B95rO3VWdhuiM6S9l9WZ5CFCFDxUK34hn6%2BVO7cf4LLJVwu6lRJAQOAvXeTSFnxh%2B5BiY7pKcEf%2FAgcnXycEvIvKDfzmJK%2F2IWFvbh0mYVPvkL5DIH1LXZiJNr70qo3u8TOSm3M%2Fc3EZtPMoTo0cGA2CEUS4y6dXQPqLEz7BnWiLN%2FOi2M84U%2BKMEviNtEwIdRZgdiE1XOpsqrYayLgVoFxbyFZgW3TnBJvwVD842nPCg6OzQTfBpzZRH82rA%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240812T181053Z&X-Amz-SignedHeaders=host&X-Amz-Expires=43200&X-Amz-Credential=ASIAWZ4O6ZMIMHVCCR77%2F20240812%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Signature=d6d74eca38bef7a3f9889b0d48a06f7d70f29214264fb56e9a920acea2c6ee7a',
                'http://bokeh.pydata.org/static/snake2.png',
                'http://bokeh.pydata.org/static/snake3D.png',
                'http://bokeh.pydata.org/static/snake4_TheRevenge.png',
                'http://bokeh.pydata.org/static/snakebite.jpg'
            ]
        )
    )

hover = HoverTool(
        tooltips="""
        <div>
            <div>
                <img
                    src="@imgs" height="300" alt="@imgs" width="900"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="2"
                ></img>
            </div>
            <div>
                <span style="font-size: 17px; font-weight: bold;">@desc</span>
                <span style="font-size: 15px; color: #966;">[$index]</span>
            </div>
            <div>
                <span style="font-size: 15px;">Location</span>
                <span style="font-size: 10px; color: #696;">($x, $y)</span>
            </div>
        </div>
        """
    )

p = figure(tools=[hover],
           title="Mouse over the dots")

p.circle('x', 'y', size=20, source=source)

# Display the plot
st.bokeh_chart(p)
