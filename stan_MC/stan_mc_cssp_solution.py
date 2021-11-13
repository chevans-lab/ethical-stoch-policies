import numpy as np


class StAnMcCsspSolution:
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