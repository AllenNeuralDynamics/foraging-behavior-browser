"""Playground for RL models of dynamic foraging
"""


import streamlit as st
import streamlit_nested_layout
from aind_behavior_gym.dynamic_foraging.task import (CoupledBlockTask,
                                                     RandomWalkTask,
                                                     UncoupledBlockTask)
from aind_dynamic_foraging_models import generative_model
from aind_dynamic_foraging_models.generative_model import ForagerCollection
from aind_dynamic_foraging_models.generative_model.params import ParamsSymbols
from aind_dynamic_foraging_models.generative_model.params.util import get_params_options
from aind_dynamic_foraging_basic_analysis import compute_foraging_efficiency

from util.streamlit import add_footnote


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
    "Compare-to-threshold": "ForagerCompareThreshold",
    "Loss counting": "ForagerLossCounting",
}

task_families = {
    "Coupled block task": "CoupledBlockTask",
    "Uncoupled block task": "UncoupledBlockTask",
    "Random walk task": "RandomWalkTask",
}


def select_agent_args(agent_args_options):
    agent_args = {}
    # Select agent parameters
    for n, arg_name in enumerate(agent_args_options.keys()):
        agent_args[arg_name] = st.selectbox(
            arg_name, 
            agent_args_options[arg_name]['options']
        )
    return agent_args

def select_params(forager):
    # Get params schema
    params_options = get_params_options(
        forager.ParamModel, 
        default_range=[-20, 20],
        para_range_override={
            "choice_kernel_relative_weight": [0.0, 2.0],
            "threshold": [-1.0, 1.0],
        })

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

def select_forager(model_family, seed=42):
    # -- Select agent --
    agent_class = getattr(generative_model, model_families[model_family])
    agent_args_options = ForagerCollection()._get_agent_kwargs_options(agent_class)
    agent_args = select_agent_args(agent_args_options)
    
    # -- Select agent parameters --
    # Initialize the agent``
    forager = agent_class(**agent_args, seed=seed)
    return forager

def select_task(task_family, reward_baiting, n_trials, seed):
    # Task parameters (hard coded for now)
    if task_family == "Coupled block task":
        block_min, block_max = st.slider("Block length range", 0, 200, [40, 80])
        block_beta = st.slider("Block beta", 0, 100, 20)
        p_reward_contrast = st.multiselect(
            "Reward contrasts",
            options=["1:1", "1:3", "1:6", "1:8"],
            default=["1:1", "1:3", "1:6", "1:8"],)
        p_reward_sum = st.slider("p_reward sum", 0.0, 1.0, 0.45)
        
        p_reward_pairs = []
        for contrast in p_reward_contrast:
            p1, p2 = contrast.split(":")
            p1, p2 = int(p1), int(p2)
            p_reward_pairs.append(
                [p1 * p_reward_sum / (p1 + p2), 
                 p2 * p_reward_sum / (p1 + p2)]
            )
            
        # Create task
        return CoupledBlockTask(
            block_min=block_min,
            block_max=block_max,
            p_reward_pairs=p_reward_pairs,
            block_beta=block_beta,
            reward_baiting=reward_baiting, 
            num_trials=n_trials, 
            seed=seed)
        
    if task_family == "Uncoupled block task":
        block_min, block_max = st.slider(
            "Block length range on each side", 0, 200, [20, 35]
            )
        rwd_prob_array = st.selectbox(
            "Reward probabilities",
            options=[[0.1, 0.5, 0.9], 
                     [0.1, 0.4, 0.7],
                     [0.1, 0.3, 0.5],
                     ],
            index=0,
        )
        max_block_tally = st.slider("max block tally", 1, 10, 4)
        persev_add = st.checkbox("anti-perseveration", value=True)
        perseverative_limit = 3
        if persev_add:
            perseverative_limit = st.slider("perseverative limit", 1, 10, 3)
        return UncoupledBlockTask(
            rwd_prob_array=rwd_prob_array,
            block_min=block_min,
            block_max=block_max,
            persev_add=persev_add,
            perseverative_limit=perseverative_limit,
            max_block_tally=max_block_tally,
            num_trials=n_trials,
            reward_baiting=reward_baiting, 
            seed=seed,
        )
        
    if task_family == "Random walk task":
        p_min, p_max = st.slider("p_reward range", 0.0, 1.0, [0.0, 1.0])
        sigma = st.slider("Random walk $\sigma$", 0.0, 1.0, 0.15)
        return RandomWalkTask(
            p_min=[p_min, p_min],
            p_max=[p_max, p_max],
            sigma=[sigma, sigma],
            mean=[0, 0],
            num_trials=n_trials,
            reward_baiting=reward_baiting, 
            seed=seed,
        )

def app():
    
    with st.sidebar:
        seed = st.number_input("Random seed", value=42)
        add_footnote()
        
    st.title("RL model playground")
    
    col0 = st.columns([1, 1])
    
    with col0[0]:
        st.markdown("#### Select agent ([🤖 aind-dynamic-foraging-models](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models/blob/develop/src/aind_dynamic_foraging_models/generative_model/foragers.py))")
        with st.expander("", expanded=True):
            col1 = st.columns([1, 2])
            with col1[0]:
                # -- Select forager family --
                agent_family = st.selectbox(
                    "Agent type", 
                    list(model_families.keys())
                )
                # -- Select forager --
                forager = select_forager(agent_family, seed=seed)
            with col1[1]:
                # -- Select forager parameters --
                params = select_params(forager)
                # Set the parameters
                forager.set_params(**params)
            
            # forager_collection = ForagerCollection()
            # all_presets = forager_collection.FORAGER_PRESETS.keys()

    with col0[1]:
        st.markdown("#### Select dynamic foraging task ([🏋️aind-behavior-gym](https://github.com/AllenNeuralDynamics/aind-behavior-gym/tree/develop/src/aind_behavior_gym/dynamic_foraging/task))")
        with st.expander("", expanded=True):
            col1 = st.columns([1, 2])
            with col1[0]:
                # -- Select task family --
                task_family = st.selectbox(
                    "Task type", 
                    list(task_families.keys()),
                    index=0,
                )
                reward_baiting = st.checkbox("Reward baiting", value=True)
                n_trials = st.slider("Number of trials", 100, 5000, 1000)
            with col1[1]:
                # -- Select task --
                task = select_task(task_family, reward_baiting, n_trials, seed)

    # -- Run the model --
    forager.perform(task)
    
    # Evaluate the foraging efficiency
    foraging_eff, foraging_eff_random_seed = compute_foraging_efficiency(
        baited=task.reward_baiting,
        choice_history=forager.get_choice_history(),
        reward_history=forager.get_reward_history(),
        p_reward=forager.get_p_reward(),
        random_number=task.random_numbers.T,
    )

    # Capture the results
    # ground_truth_params = forager.params.model_dump()
    # ground_truth_choice_prob = forager.choice_prob
    # ground_truth_q_value = forager.q_value
    # # Get the history
    # choice_history = forager.get_choice_history()
    # reward_history = forager.get_reward_history()

    # Plot the session results
    if_plot_latent = st.checkbox("Plot latent variables", value=False)
    fig, axes = forager.plot_session(if_plot_latent=if_plot_latent)
    
    col0 = st.columns([1, 0.5])
    with col0[0]:
        st.pyplot(fig)
        
        # Plot block logic
        if task_family == "Uncoupled block task":
            if_show_block_logic = st.checkbox("Show uncoupled block logic", value=False)
            if if_show_block_logic:
                fig, ax = task.plot_reward_schedule()
                ax[0].legend()
                fig.suptitle("Reward schedule")
                st.pyplot(fig)
                
    with col0[1]:
        st.write(f"#### **Foraging efficiency**:")
        st.write(f"# {foraging_eff_random_seed:.3f}")
        
    # --- Show all foragers ---
    st.markdown("---")
    st.markdown(f"### All available foragers")
    df = ForagerCollection().get_all_foragers()
    df.drop(columns=["agent_kwargs", "forager"], inplace=True)
    first_cols = ["agent_class_name", "preset_name", "params", "n_free_params"]
    df = df[
        first_cols + [col for col in df.columns if col not in first_cols]
    ]  # Reorder
    
    st.markdown(df.to_markdown(index=True))

app()