from ethical_cssp_env import EthicalCsspEnv
from typing import List, Dict, Tuple
from column_generation.RMP_solver import solve_rmp
from column_generation.SSP_random_walk import random_walks
import numpy as np
import copy
from column_generation.column_gen_cssp_solution import ColumnGenCSSPSolution
from column_generation.det_SSP_solver import optimal_deterministic_policy
import signal
import sys


def initial_solution(env: EthicalCsspEnv):
    return optimal_deterministic_policy(env)


def generate_deterministic_policies(env: EthicalCsspEnv, n_policies, banned_policies, banned_flows):
    return random_walks(env, n_policies)


def solve_cssp(env: EthicalCsspEnv, sample_size=3):
    solution = initial_solution(env)
    enforce_additional_constraints = [(False, None),
                                      (False, None),
                                      (False, None),
                                      (False, None),
                                      (False, None),
                                      (False, None)]

    while True:
        try:
            cost_vectors, policies = generate_deterministic_policies(env, sample_size, [], [])
            updated_costs = copy.copy(solution.costs)
            updated_costs = np.concatenate((updated_costs, *cost_vectors), axis=0)

            new_solution = solve_rmp(updated_costs,
                                   enforce_additional_constraints,
                                   solution.value,
                                   last[0],
                                   last[1],
                                   last[2],
                                   cost_bounds=[None, *env.secondary_cost_bounds])

            if new_solution.value < solution.value:
                solution = new_solution
                print("Found a better acceptable policy, with objective:", solution.value)
        except KeyboardInterrupt:
            print("Ending search and returning best known acceptable policy")
            return solution


if __name__ == "__main__":
    solve_cssp()