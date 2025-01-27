# Foraging behavior browser

A streamlit app for browsing foraging behavior sessions in AIND.

## The app
- The one running on ECS: https://foraging-behavior-browser.allenneuraldynamics-test.org/
    - If you see a window "Select a certificate for authentication", just click "Cancel".
- To debug, use this [Code Ocean capsule](https://codeocean.allenneuraldynamics.org/capsule/3373065/tree?cw=true)
- See also a streamlit app for ephys sessions https://foraging-ephys-browser.allenneuraldynamics-test.org/ ([Git repo](https://github.com/AllenNeuralDynamics/foraging-ephys-browser/))

## Sharing a contextual app
Starting from this [PR](https://github.com/AllenNeuralDynamics/foraging-behavior-browser/pull/25), the URL contains (part of) the session state of the app. Meaning that the user can "query" pre-set filters and plot settings in the URL. On the other hand, after interacting with the app, the URL is automatically updated to reflect the user interactions, and the user can then copy and paste the URL to share/save the new context. Essentially, this becomes a cool way of sharing a data analysis.

For example, this URL show all plots of mouse 699982

> https://foraging-behavior-browser.allenneuraldynamics-test.org/?filter_subject_id=699982&session_plot_mode=all+sessions+filtered+from+sidebar&tab_id=tab_session_inspector

![image](https://github.com/AllenNeuralDynamics/foraging-behavior-browser/assets/24734299/4389e251-1305-4a85-8936-7e5d737f8408)


and this URL will show exactly the plot below. Note the filters and plot settings are preserved.

> https://foraging-behavior-browser.allenneuraldynamics-test.org/Old_mice?filter_subject_alias=HH&filter_session=1.0&filter_session=81.0&filter_finished_trials=825.6&filter_finished_trials=1872.0&filter_foraging_eff=0.793295&filter_foraging_eff=1.2966&filter_task=coupled_block_baiting&filter_photostim_location=None&tab_id=tab_session_x_y&x_y_plot_xname=foraging_eff&x_y_plot_yname=finished_trials&x_y_plot_group_by=subject_alias&x_y_plot_if_show_dots=True&x_y_plot_if_aggr_each_group=True&x_y_plot_aggr_method_group=linear+fit&x_y_plot_if_aggr_all=False&x_y_plot_aggr_method_all=mean+%2B%2F-+sem&x_y_plot_smooth_factor=5&x_y_plot_if_use_x_quantile_group=False&x_y_plot_q_quantiles_group=20&x_y_plot_if_use_x_quantile_all=False&x_y_plot_q_quantiles_all=20&x_y_plot_dot_size=21&x_y_plot_dot_opacity=0.4&x_y_plot_line_width=3.5&auto_training_history_x_axis=session&auto_training_history_sort_by=progress_to_graduated&auto_training_history_sort_order=descending

<img width="1664" alt="image" src="https://github.com/AllenNeuralDynamics/foraging-behavior-browser/assets/24734299/2eaa7697-01cc-4eb3-bd0c-7d91c1eb64e0">

<br>
So far, theses are all supported fields in the URL query:
https://github.com/AllenNeuralDynamics/foraging-behavior-browser/blob/3f4124e98ed7aa3524b2d32c0670e8cd92ec9e41/code/Home.py#L54-L85


## Develop in Code Ocean
1. Duplicate the capsule [`foraging-behavior-browser`](https://codeocean.allenneuraldynamics.org/capsule/3373065/tree?cw=true)
2. Start a VS Code machine
3. Click "Start Debugging" or press F5
   
   <img src="https://github.com/AllenNeuralDynamics/foraging-behavior-browser/assets/24734299/0b0e70b5-d517-4d6d-a588-d5e9b3d1fb76" width=500>
4. You should see something like this in the terminal
   
   <img src="https://github.com/AllenNeuralDynamics/foraging-behavior-browser/assets/24734299/76d422bb-33f1-4387-bac1-307698426cc6" width=700>

   and a dialog like this
   
   <img src="https://github.com/AllenNeuralDynamics/foraging-behavior-browser/assets/24734299/ab87a989-c3e3-4d5b-b943-eccc19ae8d03" width=500>

   Press "Open in Browser" will initiate the app.
5. If "Open in Browser" doesn't show up, click the browser icon in `Ports - 8501 - Local Address`.

   ![image](https://github.com/AllenNeuralDynamics/foraging-behavior-browser/assets/24734299/d7665389-5bf1-469c-947d-54e48ff9fed7)

6. You can start to debug the app by adding break points etc.

