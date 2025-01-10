import plotly.io as pio

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
draw_types_quick_preview = [
    '1. Choice history',
    '3. Logistic regression (Su2022)']


def override_plotly_theme(fig, theme):
    """
    Fix the problem that simply using fig.update_layout(template=theme) doesn't work with st.plotly_chart.
    I have to use update_layout to explicitly set the theme.
    """
    
    dict_plotly_template = pio.templates[theme].layout.to_plotly_json()
    fig.update_layout(**dict_plotly_template)  # First apply the plotly official theme
    
    fig.update_layout(font_family="Arial")  # Add user-defined styling
    
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', showgrid=True, zeroline=True,
                     ticks = "outside", tickcolor='black', ticklen=10, tickwidth=2, ticksuffix=' ')

    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', showgrid=True, zeroline=True,
                     ticks = "outside", tickcolor='black', ticklen=10, tickwidth=2, ticksuffix=' ',
                     title_standoff=40,
                     )

    return