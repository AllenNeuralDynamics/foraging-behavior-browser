import plotly.io as pio
pio.json.config.default_engine = "orjson"

# Setting up layout for each session
draw_type_layout_definition = [
    [1],  # columns in the first row
    [1],
    [1, 1],  # columns in the second row
    [1, 1],
]

logistic_regression_models = ["Su2022", "Bari2019", "Hattori2019", "Miller2021"]

draw_type_mapper_session_level = {
    "1. Choice history": (
        "choice_history",  # prefix
        (0, 0),  # location (row_idx, column_idx)
        dict(),
    ),
    "2. Lick analysis": (
        "lick_analysis",
        (1, 0),  # location (row_idx, column_idx)
        dict(),
    ),
    **{
        f"{n + 3}. Logistic regression ({model})": (
            f"logistic_regression_{model}",  # prefix
            (2 + int(n / 2), n % 2),  # location (row_idx, column_idx)
            dict(),
        )
        for n, model in enumerate(logistic_regression_models)
    },
}

# For quick preview
draw_types_quick_preview = ["1. Choice history", "3. Logistic regression (Su2022)"]


# For plotly styling
PLOTLY_FIG_DEFAULT = dict(
        font_family="Arial",
    )
PLOTLY_AXIS_DEFAULT = dict(
        showline=True,
        linewidth=2,
        linecolor="black",
        showgrid=True,
        gridcolor="lightgray",
        griddash="solid",
        minor_showgrid=False,
        minor_gridcolor="lightgray",
        minor_griddash="solid",
        zeroline=True,
        ticks="outside",
        tickcolor="black",
        ticklen=7,
        tickwidth=2,
        ticksuffix=" ",
        tickfont=dict(
            family="Arial",
            color="black",
        ),
    )

def override_plotly_theme(
    fig,
    theme="simple_white",
    fig_specs=PLOTLY_FIG_DEFAULT,
    axis_specs=PLOTLY_AXIS_DEFAULT,
    font_size_scale=1.0,
):
    """
    Fix the problem that simply using fig.update_layout(template=theme) doesn't work with st.plotly_chart.
    I have to use update_layout to explicitly set the theme.
    """

    dict_plotly_template = pio.templates[theme].layout.to_plotly_json()
    fig.update_layout(**dict_plotly_template)  # First apply the plotly official theme

    # Apply settings to all x-axes
    for axis in fig.layout:
        if axis.startswith('xaxis') or axis.startswith('yaxis'):
            fig.layout[axis].update(axis_specs)
            fig.layout[axis].update(
                tickfont_size=20 * font_size_scale, 
                title_font_size=22 * font_size_scale,
            )
        if axis.startswith("yaxis"):
            fig.layout[axis].update(title_standoff=10 * font_size_scale)

    fig.update_layout(**fig_specs)  # Apply settings to the entire figure
    
    # Customize the font of subplot titles
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(
            family="Arial",  # Font family
            size=20 * font_size_scale,    # Font size
            color="black"     # Font color
        )

    fig.update_layout(
        font_size=22 * font_size_scale,
        hoverlabel_font_size=17 * font_size_scale,
        legend_font_size=20 * font_size_scale,
        margin=dict(
            l=130 * font_size_scale,
            r=50 * font_size_scale,
            b=130 * font_size_scale,
            t=100 * font_size_scale,
        ),
    )
    return
