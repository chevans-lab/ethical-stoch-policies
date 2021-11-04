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


def solve_cssp(env: EthicalCsspEnv, sample_size=10, n_iterations=100):
    solution = initial_solution(env)
    enforce_additional_constraints = [(True, 10),
                                      (False, 0.5),
                                      (False, 0),
                                      (False, 0),
                                      (True, 10),
                                      (False, 0)]

    track_worst = enforce_additional_constraints[0][0] or enforce_additional_constraints[1][0] or enforce_additional_constraints[2][0] or enforce_additional_constraints[3][0]
    track_cvar = enforce_additional_constraints[4][0] or enforce_additional_constraints[5][0]

    plot_data = {}
    plot_data["Value"] = np.empty(n_iterations)
    plot_data["Value"][0] = solution.value

    if track_worst:
        plot_data["Worst"] = np.empty(n_iterations)
        plot_data["Worst"][0] = solution.value
    if track_cvar:
        plot_data["CVaR"] = np.empty(n_iterations)
        plot_data["CVaR"][0] = solution.value

    for i in range(1, n_iterations):
        try:
            print(i)
            cost_vectors, policies = generate_deterministic_policies(env, sample_size, [], [])

            new_solution = copy.deepcopy(solution)
            new_solution.costs = np.concatenate((new_solution.costs, cost_vectors), axis=0)
            new_solution.policies = np.concatenate((new_solution.policies, policies))

            new_solution = solve_rmp(env, new_solution, enforce_additional_constraints, alpha=0.95)

            if new_solution.value < solution.value:
                solution = new_solution
                print("Found a better acceptable policy, with objective:", solution.value)

            plot_data["Value"][i] = solution.value
            if track_worst:
                plot_data["Worst"][i] = solution.worst_case_value
            if track_cvar:
                plot_data["CVaR"][i] = solution.cvar

        except KeyboardInterrupt:
            print("Ending search and returning best known acceptable policy")
            return solution, plot_data

    print("Ending search and returning best known acceptable policy")
    return solution, plot_data
