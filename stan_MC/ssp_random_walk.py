from env.mc_cssp_env import MorallyConsequentialCsspEnv, State, Action, ActionName

from typing import List, Tuple, Dict
from random import randrange
import numpy as np

"""
Author: Charles Evans
Email: u6942700@anu.edu.au
This is my own work, and forms part of my artefact contribution for COMP3770, Semester 1, 2021.
"""


def random_walk(env: MorallyConsequentialCsspEnv, n_policies: int):

    """
    Generates a collection of random deterministic policies for the C-SSP instance `env' using a random walk approach.
    Treats the C-SSP as a flow network (as in the dual LP), and:
    - Initiates a unit of flow at s0
    - Randomly assigns an applicable action when it meets a new state
    - Distributes flow to successors according to the random action
    Until all flow has reached goal states.

    Algorithm assumes an acyclic C-SSP.
    Returned policies are not guaranteed to feasible for the C-SSP, either collectively or in their own right.

    Args:
        env: The morally consequential C-SSP instance
        n_policies: Number of deterministic policies to sample and return

    Returns:
        np.ndarray (2D float array), np.ndarray (1D array of Dict[int, Dict[str, float]] objects)
    """

    policies = []  # ordered collection of generated policies
    cost_vectors = []  # ordered collection of expected cost vectors of generated policies

    while n_policies > 0:
        # Dictionary of flow values for each (state, action) pair
        flow: Dict[int, Dict[ActionName, float]] = {}
        # Policy dictionary. Each entry will be the ceil. of its analog in the flow dictionary
        policy: Dict[int, Dict[ActionName, float]] = {}
        # Queue of open states, and the flow reaching them. Initialised with the initial state and an inflow of 1.
        open_states: List[Tuple[State, float]] = [(env.initial_state, 1)]
        # State -> Action mapping. Will be established the first time a state is visited
        established_actions: Dict[int, Action] = {}

        while open_states:
            s, in_flow = open_states.pop(0)
            a: Action
            if s.id in established_actions:
                # We have visited this state already, so follow already-chosen action and increment flow of that pairing
                a = established_actions[s.id]
                flow[s.id][a.name] += in_flow
            else:
                flow[s.id] = {}
                policy[s.id] = {}
                applicable_actions = env.applicable_actions(s)
                for action in env.applicable_actions(s):
                    policy[s.id][action.name] = 0
                # Randomly sample an applicable action to be taken at s
                a = applicable_actions[randrange(len(applicable_actions))]
                flow[s.id][a.name] = in_flow
                policy[s.id][a.name] = 1
                established_actions[s.id] = a

            # Open all successor states with their proportion of the inflow given the transition probabilities
            for s_, prob in env.transition_probabilities(s, a).get_values():
                if not env.terminal_state(s_):
                    open_states.append((s_, prob * in_flow))

        # Make flow 0 for all unvisited valid state-action pairs
        for s in env.state_space:
            for a in env.applicable_actions(s):
                if s.id not in flow:
                    flow[s.id] = {}
                if a.name not in flow[s.id]:
                    flow[s.id][a.name] = 0

        # Calculates expected cost vector for the generated policy
        costs = np.array([flow[s.id][a.name] * np.array(env.transition_costs(s, a))
                         for s in env.state_space for a in env.applicable_actions(s) if s.id in flow])
        summated_costs = np.sum(costs, axis=0)

        policies.append(policy)  # store policy
        cost_vectors.append(summated_costs)  # store policy's expected cost vector
        n_policies -= 1

    # Return sampled policies as a 2D array
    return np.array(cost_vectors), np.array(policies)
