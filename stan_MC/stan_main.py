from env.mc_cssp_env import MorallyConsequentialCsspEnv
from stan_MC.augmented_rmp import solve_rmp
from stan_MC.ssp_random_walk import random_walk
from stan_MC.det_ssp_solver import optimal_deterministic_policy

from typing import Dict
import numpy as np
import copy
import time

"""
Author: Charles Evans
Email: u6942700@anu.edu.au
This is my own work, and forms part of my artefact contribution for COMP3770, Semester 1, 2021.
"""


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


def solve_cssp(env: MorallyConsequentialCsspEnv,
               constraint_params:
               Dict[str, float],
               sample_size: int,
               iterations: int,
               store_plot_data=False):

    """
    Solves a morally consquential C-SSP using StAn-MC. Returns a tuple:
    - the policy object
    - a dictionary of data to plot the solution evolution (if requested). Returns None for this element otherwise.
    - the percentage by which the solver was able to decrease the expected cost of the initial (deterministic) solution
    - the time taken to produce the solution, in seconds
    
    
    Args:
        env: The morally consequential C-SSP instance
        constraint_params: A string -> float mapping for parameterising the constraints we want to enforce.
            -- If "bound_wcv" is provided, it maps to the upper bound on worst-case value that should be enforced
            -- If "bound_ewd" is provided, it maps to the upper bound on difference between expected and worst-case value that should be enforced
            -- If "tradeoff_wcv" is provided, it maps to the weighting to give to the worst-case value increase in the tradeoff constraint.
            -- If "tradeoff_cvar" is provided, it maps to the weighting to give to the conditional value at risk increase in the tradeoff constraint.
        sample_size: Batch size of deterministic policies to randomly sample at each iteration of StAn-MC.
        iterations: Number of policy improvement iterations for StAn-MC to perform.
        store_plot_data: Boolean configuring whether to store the expected value and other metrics about the policy 
            at each iteration, for use in plotting algorithm performance

    Returns:
        Tuple[StAnMcCsspSolution, Dict[str, np.ndarray], float, float]
    """

    plot_data = None
    if store_plot_data:
        # Creating data arrays to store data for result plotting
        plot_data = {"Value": np.empty(iterations)}
        if "tradeoff_wcv" in constraint_params or "bound_wcv" in constraint_params:
            plot_data["Worst"] = np.empty(iterations)
        if "bound_ewd" in constraint_params:
            plot_data["Expected_Worst_Diff"] = np.empty(iterations)
        if "tradeoff_cvar" in constraint_params:
            plot_data["CVaR"] = np.empty(iterations)

    # Produce initial feasible & acceptable solution
    start = time.time()
    solution = initial_solution(env)
    initial_solution_value = solution.value
    print(f"Initial feasible and acceptable policy found: expected primary cost = {initial_solution_value}")
    print(f"Initial solution to undergo {iterations - 1} potential improvement iterations:")

    if store_plot_data:
        plot_data["Value"][0] = solution.value
        if "tradeoff_wcv" in constraint_params or "bound_wcv" in constraint_params:
            plot_data["Worst"][0] = solution.value
        if "bound_ewd" in constraint_params:
            plot_data["Expected_Worst_Diff"][0] = 0
        if "tradeoff_cvar" in constraint_params:
            plot_data["CVaR"][0] = solution.value

    for i in range(1, iterations):
        # Randomly sample <sample_size> deterministic policies
        cost_vectors, policies = generate_deterministic_policies(env, sample_size)

        # Generate a new solution by augmenting the set of active policies (and associated expected costs)
        new_solution = copy.deepcopy(solution)
        new_solution.costs = np.concatenate((new_solution.costs, cost_vectors), axis=0)
        new_solution.policies = np.concatenate((new_solution.policies, policies))

        # Optimise the probability distribution of the new solution (must be feasible and acceptable)
        new_solution = solve_rmp(env, new_solution, constraint_params)

        # Updating current best solution if an improvement in objective has been achieved
        if new_solution.value < solution.value:
            solution = new_solution
            print(f"Iteration {i}: New feasible and acceptable policy found with lower expected primary cost: {solution.value}")

        if store_plot_data:
            # Recording datapoints of expected value and disadvantage measures as of i-th iteration
            plot_data["Value"][i] = solution.value
            if "tradeoff_wcv" in constraint_params or "bound_wcv" in constraint_params:
                plot_data["Worst"][i] = solution.worst_case_value
            if "bound_ewd" in constraint_params:
                plot_data["Expected_Worst_Diff"][i] = solution.worst_case_value - solution.value
            if "tradeoff_cvar" in constraint_params:
                plot_data["CVaR"][i] = solution.cvar

    print("Ending search and returning best known feasible and acceptable policy")
    print("---------------------------------------------------------------------")
    print()
    # Calculating the percentage improvement in expected primary cost of returned solution vs. initial solution
    percentage_reduction = ((initial_solution_value - solution.value) * 100) / initial_solution_value

    return solution, plot_data, percentage_reduction, (time.time() - start)
