from typing import List, Dict
import numpy as np


class ColumnGenCSSPSolution:
    probabilities: List[float]
    costs: np.ndarray
    policies: List[Dict[int, Dict[str, float]]]
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

    def get_policy_value(self):
        v = 0
        for i in range(len(self.probabilities)):
            v += self.probabilities[i] * self.costs[i, 0]
        return v