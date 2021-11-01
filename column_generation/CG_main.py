from ethical_cssp_env import EthicalCsspEnv
from column_generation.RMP_solver import solve_rmp
from column_generation.SSP_random_walk import random_walk
import numpy as np
import copy
from column_generation.det_SSP_solver import optimal_deterministic_policy


def initial_solution(env: EthicalCsspEnv):
    return optimal_deterministic_policy(env)


def generate_deterministic_policies(env: EthicalCsspEnv, n_policies, banned_policies, banned_flows):
    return random_walk(env, n_policies)


def solve_cssp(env: EthicalCsspEnv, sample_size=10):
    solution = initial_solution(env)
    enforce_additional_constraints = [(False, 0),
                                      (True, 1),
                                      (False, 0),
                                      (False, 0),
                                      (False, 0),
                                      (False, 0)]

    x = 50
    while x > 0:
        try:
            cost_vectors, policies = generate_deterministic_policies(env, sample_size, [], [])

            new_solution = copy.deepcopy(solution)
            new_solution.costs = np.concatenate((new_solution.costs, cost_vectors), axis=0)
            new_solution.policies = np.concatenate((new_solution.policies, policies))

            new_solution = solve_rmp(env, new_solution, enforce_additional_constraints, alpha=0.9)

            if new_solution.value < solution.value:
                solution = new_solution
                print("Found a better acceptable policy, with objective:", solution.value)
            x -= 1
        except KeyboardInterrupt:
            print("Ending search and returning best known acceptable policy")
            return solution

    return solution
