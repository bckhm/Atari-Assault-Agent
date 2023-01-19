import base64
import random
from itertools import zip_longest

import imageio
import IPython
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import tensorflow as tf
from statsmodels.iolib.table import SimpleTable

def display_table(initial_state, action, next_state, reward, done):
    """
    Displays a table containing the initial state, action, next state, reward, and done
    values from Gym's Bipedal Walker Environment.


    Args:
        initial_state (numpy.ndarray):
            The initial state vector returned when resetting the Bipedal Walker
            environment, i.e the value returned by the env.reset() method.
        action (list):
            The action taken by the agent. In the Bipedal Walker environment, actions are
            represented by lists in the closed interval [-1,1] for each element. The elements correspond to:
                - Hip 1
                - Knee 1
                - Hip 2
                - Knee 2 
        next_state (numpy.ndarray):
            The state vector returned by the environment after the agent
            takes an action, i.e the observation returned after running a single time
            step of the environment's dynamics using env.step(action).
        reward (numpy.float64):
            The reward returned by the  environment after the agent takes an
            action, i.e the reward returned after running a single time step of the
            environment's dynamics using env.step(action).
        done (bool):
            The done value returned by the  environment after the agent
            takes an action, i.e the done value returned after running a single time
            step of the environment's dynamics using env.step(action).
    
    Returns:
        table (statsmodels.iolib.table.SimpleTable):
            A table object containing the initial_state, action, next_state, reward,
            and done values. This will result in the table being displayed in the
            Jupyter Notebook.
    """

    # Do not use column headers.
    column_headers = None

    # Display all floating point numbers rounded to 3 decimal places.
    with np.printoptions(formatter={"float": "{:.3f}".format}):
        table_info = [
            ("Initial State:", [f"{initial_state}"]),
            ("Hip1 Action:", action[0]),
            ("Knee1 Action:", action[1]),
            ("Hip2 Action:", action[2]),
            ("Knee2 Action:", action[3]),
            ("Next State:", [f"{next_state}"]),
            ("Reward Received:", [f"{reward:.3f}"]),
            ("Episode Terminated:", [f"{done}"]),
        ]

    # Generate table.
    row_labels, data = zip_longest(*table_info)
    table = SimpleTable(data, column_headers, row_labels)

    return table


def update_target_network(q_network, target_q_network, TAU):
    """
    Updates the weights of the target Q-Network using a soft update.
    
    The weights of the target_q_network are updated using the soft update rule:
    
                    w_target = (TAU * w) + (1 - TAU) * w_target
    
    where w_target are the weights of the target_q_network, TAU is the soft update
    parameter, and w are the weights of the q_network.
    
    Args:
        q_network (tf.keras.Sequential): 
            The Q-Network. 
        target_q_network (tf.keras.Sequential):
            The Target Q-Network.
        TAU (float):
            Soft-update parameter
    """

    for target_weights, q_net_weights in zip(
        target_q_network.weights, q_network.weights
    ):
        target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)
        
        
def get_action(q_values, epsilon=0.0):
    """
    Returns an action using an ε-greedy policy.

    This function will return an action according to the following rules:
        - With probability epsilon, it will return an action chosen at random.
        - With probability (1 - epsilon), it will return the action that yields the
        maximum Q value in q_values.
    
    Args:
        q_values (tf.Tensor):
            The Q values returned by the Q-Network. For the Bipedal Walker environment
            this TensorFlow Tensor should have a shape of [4, 4] and its elements should
            have dtype=tf.float32. 
        epsilon (float):
            The current value of epsilon.

    Returns:
       An action (np.array) with shape (4,). For the Bipedal Walker environment, actions are
       represented by np.arrays with each element in the closed continuous interval [-1, 1].
    """
     # Any random number x has ||(x-epsilon)|| probability of happening, in which case we 
        # Choose the most optimal (highest Q(s,a)) acgtion
        # Else, we choose random actions
    if random.random() > epsilon:
        return q_values.max()
    
    else:
        return np.random.uniform(-1, 1, size=4)
    
    
        