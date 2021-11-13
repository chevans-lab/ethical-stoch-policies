import numpy as np

"""
Author: Charles Evans
Email: u6942700@anu.edu.au
This is my own work, and forms part of my artefact contribution for COMP3770, Semester 1, 2021.
"""

class StAnMcCsspSolution:
    """
    StAn-MC C-SSP Solution (StAnMcCsspSolution).
    Represents a stochastic policy for a morally consequential C-SSP in a 'concentrated' representation.
    The probabilities, costs and policies fields are numpy arrays, all ordered in the same way,
    where the i-th entries of the former two arrays represent the probability of sampling the i-th (deterministic) policy in policies,
    and its expected cost vector, respectively.

    The value field represents the overall expected primary cost, and worst_case_value and cvar represent
    the expected value of the worst deterministic policy in the mix, and the conditional value-at-risk of the mix.
    """
    probabilities: np.ndarray
    costs: np.ndarray
    policies: np.ndarray
    value: float
    worst_case_value: float
    cvar: float

    def __init__(self, probabilities, costs, policies, value, worst_case_value, cvar):
        self.probabilities = probabilities
        self.costs = costs
        self.policies = policies
        self.value = value
        self.worst_case_value = worst_case_value
        self.cvar = cvar
