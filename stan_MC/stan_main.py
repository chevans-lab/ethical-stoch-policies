from env.ethical_cssp_env import MorallyConsequentialCsspEnv
from stan_MC.augmented_rmp import solve_rmp
from stan_MC.ssp_random_walk import random_walk
from stan_MC.det_SSP_solver import optimal_deterministic_policy

from typing import Dict
import numpy as np
import copy


def initial_solution(env: MorallyConsequentialCsspEnv):
    """
    Generates an initial solution to the morally consequential C-SSP. This will be the optimal feasible deterministic policy,
    represented in the form of a StAnMcCsspSolution object.

    Args:
        env: The morally consequential C-SSP instance
    Returns:
        StAnMcCsspSolution
    """
    return optimal_deterministic_policy(env)


def generate_deterministic_policies(env: MorallyConsequentialCsspEnv, n_policies):
    """
    Randomly samples a batch of deterministic policies for some (acyclic) C-SSP. Policies may be infeasible.

    Args:
        env: The morally consequential C-SSP instance
        n_policies: The number of deterministic policies to sample

    Returns:

    """
    return random_walk(env, n_policies)


def solve_cssp(env: MorallyConsequentialCsspEnv, constraint_params: Dict[str, float], sample_size: int, iterations: int):
    enforce_additional_constraints = [(False, 10),
                                      (False, 0.5),
                                      (False, 0),
                                      (False, 0),
                                      (True, 1.4),
                                      (False, 0)]

    track_wcv = "tradeoff_wcv" in constraint_params or "bound_wcv" in constraint_params
    track_ewd = "bound_ewd" in constraint_params
    track_cvar = "tradeoff_cvar" in constraint_params
    plot_data = {}
    plot_data["Value"] = np.empty(iterations)
    if track_wcv:
        plot_data["Worst"] = np.empty(iterations)
    if track_ewd:
        plot_data["Expected_Worst_Diff"] = np.empty(iterations)
    if track_cvar:
        plot_data["CVaR"] = np.empty(iterations)

    # Produce initial feasible & acceptable solution
    solution = initial_solution(env)

    plot_data["Value"][0] = solution.value
    if track_wcv:
        plot_data["Worst"][0] = solution.value
    if track_ewd:
        plot_data["Expected_Worst_Diff"][0] = 0
    if track_cvar:
        plot_data["CVaR"][0] = solution.value

    for i in range(1, iterations):
        try:
            # Randomly sample <sample_size> deterministic policies
            cost_vectors, policies = generate_deterministic_policies(env, sample_size)

            new_solution = copy.deepcopy(solution)
            new_solution.costs = np.concatenate((new_solution.costs, cost_vectors), axis=0)
            new_solution.policies = np.concatenate((new_solution.policies, policies))

            new_solution = solve_rmp(env, new_solution, constraint_params)

            if new_solution.value < solution.value:
                solution = new_solution
                print("Found a better acceptable policy, with objective:", solution.value)

            plot_data["Value"][i] = solution.value
            if track_wcv:
                plot_data["Worst"][i] = solution.worst_case_value
            if track_ewd:
                plot_data["Expected_Worst_Diff"][i] = solution.worst_case_value - solution.value
            if track_cvar:
                plot_data["CVaR"][i] = solution.cvar

        except KeyboardInterrupt:
            print("Ending search and returning best known acceptable policy")
            return solution, plot_data

    print("Ending search and returning best known acceptable policy")
    return solution, plot_data
