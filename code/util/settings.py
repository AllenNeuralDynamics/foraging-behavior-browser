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
