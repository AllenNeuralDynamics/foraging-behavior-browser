"""Playground for RL models of dynamic foraging
"""

import streamlit as st
import streamlit_nested_layout
from typing import get_type_hints, _LiteralGenericAlias
import inspect

from aind_behavior_gym.dynamic_foraging.task import CoupledBlockTask
from aind_dynamic_foraging_models import generative_model
from aind_dynamic_foraging_models.generative_model import ForagerCollection

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

model_families = {
    "Q-learning": "ForagerQLearning",
    "Loss counting": "ForagerLossCounting"
}

def _get_agent_args(agent_class):
    type_hints = get_type_hints(agent_class.__init__)
    signature = inspect.signature(agent_class.__init__)

    agent_args_options = {}
    for arg, type_hint in type_hints.items():
        if isinstance(type_hint, _LiteralGenericAlias):  # Check if the type hint is a Literal
            # Get options
            literal_values = type_hint.__args__
            default_value = signature.parameters[arg].default
            
            agent_args_options[arg] = {'options': literal_values, 'default': default_value}
    return agent_args_options

def get_arg_params(agent_args_options):
    agent_args = {}
    # Select agent parameters
    for n, arg_name in enumerate(agent_args_options.keys()):
        agent_args[arg_name] = st.selectbox(
            arg_name, 
            agent_args_options[arg_name]['options']
        )
    return agent_args


def app():
    
    # A dropdown to select model family
    model_family = st.selectbox("Select model family", list(model_families.keys()))
    
    agent_class = getattr(generative_model, model_families[model_family])
    agent_args_options = _get_agent_args(agent_class)
    agent_args = get_arg_params(agent_args_options)
    
    # Initialize the model
    forager_collection = ForagerCollection()
    all_presets = forager_collection.FORAGER_PRESETS.keys()
    
    forager = forager_collection.get_preset_forager("Hattori2019", seed=42)
    
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