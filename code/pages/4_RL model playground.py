"""Playground for RL models of dynamic foraging
"""

import streamlit as st
import streamlit_nested_layout
from typing import get_type_hints, _LiteralGenericAlias
import inspect

from aind_behavior_gym.dynamic_foraging.task import CoupledBlockTask
from aind_dynamic_foraging_models import generative_model
from aind_dynamic_foraging_models.generative_model import ForagerCollection
from aind_dynamic_foraging_models.generative_model.params import ParamsSymbols

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

para_range_override = {
    "biasL": [-5.0, 5.0],
    "choice_kernel_relative_weight": [0.0, 2.0],
}

def _get_agent_args_options(agent_class):
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


def _get_params_options(param_model):
    # Get the schema
    params_schema = param_model.model_json_schema()["properties"]

    # Extract ge and le constraints
    param_options = {}

    for para_name, para_field in params_schema.items():
        default = para_field.get("default", None)
        para_desc = para_field.get("description", "")
        
        if para_name in para_range_override:
            para_range = para_range_override[para_name]
        else:  # Get from pydantic schema
            para_range = [-20, 20]  # Default range
            # Override the range if specified
            if "minimum" in para_field:
                para_range[0] = para_field["minimum"]
            if "maximum" in para_field:
                para_range[1] = para_field["maximum"]
            para_range = [type(default)(x) for x in para_range]
            
        param_options[para_name] = dict(
            para_range=para_range,
            para_default=default,
            para_symbol=ParamsSymbols[para_name],
            para_desc=para_desc,
        )
    return param_options

def select_agent_args(agent_args_options):
    agent_args = {}
    # Select agent parameters
    for n, arg_name in enumerate(agent_args_options.keys()):
        agent_args[arg_name] = st.selectbox(
            arg_name, 
            agent_args_options[arg_name]['options']
        )
    return agent_args

def select_params(params_options):
    params = {}
    # Select agent parameters
    for n, para_name in enumerate(params_options.keys()):
        para_range = params_options[para_name]['para_range']
        para_default = params_options[para_name]['para_default']
        para_symbol = params_options[para_name]['para_symbol']
        para_desc = params_options[para_name].get('para_desc', '')
        
        if para_range[0] == para_range[1]:  # Fixed parameter
            params[para_name] = para_range[0]
            st.markdown(f"{para_symbol} ({para_name}) is fixed at {para_range[0]}")
        else:
            params[para_name] = st.slider(
                f"{para_symbol} ({para_name})", 
                para_range[0], para_range[1], 
                para_default
            )
    return params

def set_forager(model_family, seed=42):
    # -- Select agent --
    agent_class = getattr(generative_model, model_families[model_family])
    agent_args_options = _get_agent_args_options(agent_class)
    agent_args = select_agent_args(agent_args_options)
    
    # -- Select agent parameters --
    # Initialize the agent
    forager = agent_class(**agent_args, seed=seed)
    # Get params options
    params_options = _get_params_options(forager.ParamModel)
    params = select_params(params_options)
    # Set the parameters
    forager.set_params(**params)
    return forager

def app():
    
    with st.sidebar:
        seed = st.number_input("Random seed", value=42)
        
    
    # -- Select forager family --
    agent_family = st.selectbox("Select model family", list(model_families.keys()))
    
    # -- Select forager --
    forager = set_forager(agent_family, seed=seed)

    # forager_collection = ForagerCollection()
    # all_presets = forager_collection.FORAGER_PRESETS.keys()

    # Create the task environment
    task = CoupledBlockTask(reward_baiting=True, num_trials=1000, seed=seed) 

    # -- Run the model --
    forager.perform(task)
    
    if_plot_latent = st.checkbox("Plot latent variables", value=False)

    # Capture the results
    ground_truth_params = forager.params.model_dump()
    ground_truth_choice_prob = forager.choice_prob
    ground_truth_q_value = forager.q_value
    # Get the history
    choice_history = forager.get_choice_history()
    reward_history = forager.get_reward_history()

    # Plot the session results
    fig, axes = forager.plot_session(if_plot_latent=if_plot_latent)  
    st.pyplot(fig)

app()