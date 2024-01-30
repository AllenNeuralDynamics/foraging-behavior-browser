# Foraging behavior browser

A streamlit app for browsing foraging behavior sessions in AIND.

## The app
- The one running on Streamlit Community Cloud: https://foraging-behavior-browser.streamlit.app/
- For a better experience, click the Code Ocean link shown at the top of the app
  <img width="729" alt="image" src="https://github.com/AllenNeuralDynamics/foraging-behavior-browser/assets/24734299/01dfe9d2-a98f-46a2-a436-982d087d6b0d">
- See also a streamlit app for ephys sessions https://github.com/AllenNeuralDynamics/foraging-ephys-browser/

## Sharing a contextual app
Starting from this [PR](https://github.com/AllenNeuralDynamics/foraging-behavior-browser/pull/25), the URL contains (part of) the session state of the app. Meaning that the user can "query" pre-set filters and plot settings in the URL. On the other hand, after interacting with the app, the URL is automatically updated to reflect the user interactions, and the user can then copy and paste the URL to share/save the new context. Essentially, this becomes a cool way of sharing a data analysis.

For example, this URL will show exactly the plot below. Note the filters and plot settings are preserved.

> https://foraging-behavior-browser.streamlit.app/Old_mice?filter_h2o=HH&filter_session=1.0&filter_session=81.0&filter_finished_trials=825.6&filter_finished_trials=1872.0&filter_foraging_eff=0.793295&filter_foraging_eff=1.2966&filter_task=coupled_block_baiting&filter_photostim_location=None&tab_id=tab_session_x_y&x_y_plot_xname=foraging_eff&x_y_plot_yname=finished_trials&x_y_plot_group_by=h2o&x_y_plot_if_show_dots=True&x_y_plot_if_aggr_each_group=True&x_y_plot_aggr_method_group=linear+fit&x_y_plot_if_aggr_all=False&x_y_plot_aggr_method_all=mean+%2B%2F-+sem&x_y_plot_smooth_factor=5&x_y_plot_if_use_x_quantile_group=False&x_y_plot_q_quantiles_group=20&x_y_plot_if_use_x_quantile_all=False&x_y_plot_q_quantiles_all=20&x_y_plot_dot_size=21&x_y_plot_dot_opacity=0.4&x_y_plot_line_width=3.5&auto_training_history_x_axis=session&auto_training_history_sort_by=progress_to_graduated&auto_training_history_sort_order=descending

<img width="1664" alt="image" src="https://github.com/AllenNeuralDynamics/foraging-behavior-browser/assets/24734299/2eaa7697-01cc-4eb3-bd0c-7d91c1eb64e0">

<br><br>
So far, theses are all supported fields in the URL query:
https://github.com/AllenNeuralDynamics/foraging-behavior-browser/blob/3f4124e98ed7aa3524b2d32c0670e8cd92ec9e41/code/Home.py#L54-L85


