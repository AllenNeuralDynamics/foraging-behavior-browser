"""Playground for RL models of dynamic foraging
"""

import streamlit as st
import streamlit_nested_layout

from aind_behavior_gym.dynamic_foraging.task import CoupledBlockTask
from aind_dynamic_foraging_models.generative_model import (
    ForagerQLearning, ForagerLossCounting, ForagerCollection
)

try:
    st.set_page_config(layout="wide", 
                    page_title='Foraging behavior browser',
                    page_icon=':mouse2:',
                        menu_items={
                        'Report a bug': "https://github.com/hanhou/foraging-behavior-browser/issues",
                        'About': "Github repo: https://github.com/hanhou/foraging-behavior-browser/"
                        }
                    )
except:
    pass

def app():
    # Initialize the model
    forager = ForagerCollection().get_preset_forager("Hattori2019", seed=42)
    forager.set_params(
        softmax_inverse_temperature=5,
        biasL=0,
    )

    # Create the task environment
    task = CoupledBlockTask(reward_baiting=True, num_trials=1000, seed=42) 

    # Run the model
    forager.perform(task)

    # Capture the results
    ground_truth_params = forager.params.model_dump()
    ground_truth_choice_prob = forager.choice_prob
    ground_truth_q_value = forager.q_value
    # Get the history
    choice_history = forager.get_choice_history()
    reward_history = forager.get_reward_history()

    # Plot the session results
    fig, axes = forager.plot_session(if_plot_latent=True)  
    st.pyplot(fig)

app()