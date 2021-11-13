from stan_MC.stan_mc_cssp_solution import StAnMcCsspSolution
from env.mc_cssp_env import MorallyConsequentialCsspEnv

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from typing import Dict

"""
Author: Charles Evans
Email: u6942700@anu.edu.au
This is my own work, and forms part of my artefact contribution for COMP3770, Semester 1, 2021.
"""

def solve_rmp(env: MorallyConsequentialCsspEnv,
              unopt_solution: StAnMcCsspSolution,
              constraint_params: Dict[str, float]):

    """
    Solves the RMP proposed in "Optimal and Heuristic Approaches for Constrained Flight Planning under Weather Uncertainty", Geisser et al. (2020)
    *augmented by* our moral acceptability constraints. Returns an optimised version of an unoptimised policy `unopt_solution`
    (re-solves for the optimal (feasible and acceptable) probability distribution over the available det. policies)

    Args:
        env: The morally consequential C-SSP instance
        unopt_solution: an unoptimised solution to the C-SSP instance. new deterministic policies have been added to its usable set,
            but the probability distribution over them needs to be re-optimised (which is the job of this function)
        constraint_params: A string -> float mapping for parameterising the constraints we want to enforce.
            -- If "bound_wcv" is provided, it maps to the upper bound on worst-case value that should be enforced
            -- If "bound_ewd" is provided, it maps to the upper bound on difference between expected and worst-case value that should be enforced
            -- If "tradeoff_wcv" is provided, it maps to the weighting to give to the worst-case value increase in the tradeoff constraint.
            -- If "tradeoff_cvar" is provided, it maps to the weighting to give to the conditional value at risk increase in the tradeoff constraint.
    Returns:
        StAnMcCsspSolution
    """

    solver_env = gp.Env(empty=True)
    solver_env.setParam("OutputFlag", 0)
    solver_env.start()
    m = gp.Model("augmented_rmp", env=solver_env)

    mc_cost_index = 0  # index of C-SSP cost function that is morally consequential (always the 0th in our formulation)
    alpha = 0.9  # preconfiguring the alpha hyperparameter for CVaR tradeoff constraint; see report for more info
    active_policy_epsilon = 10e-6  # A det. policy sampled with this probability or higher is considered `active'
    tradeoff_relaxation = 10e-5  # Used to relax the tradeoff constraint by a small constant (so the last solution will always be feasible even with minor floating point errors)

    num_policies = unopt_solution.costs.shape[0]  # number of available deterministic policies in the solution

    # Sorting the policies and their expected cost vectors in order of increasing primary (MC) expected cost order
    increasing_cost_order = unopt_solution.costs[:, mc_cost_index].argsort()
    unopt_solution.costs = unopt_solution.costs[increasing_cost_order]
    unopt_solution.policies = unopt_solution.policies[increasing_cost_order]

    # Finding min and max primary (MC) expected costs among the available det. policies
    max_cost = unopt_solution.costs[num_policies - 1, mc_cost_index]
    min_cost = unopt_solution.costs[0, mc_cost_index]

    # Probability decision variables (i-th var. gives probability of sampling i-th deterministic policy)
    probabilities = m.addVars(range(num_policies), name="probability", lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)
    m.addConstr(probabilities.sum() == 1.0)  # enforcing normalised probabilities

    # Enforcing feasible solutions only (C-SSP secondary cost bounds)
    for j in range(env.num_secondary_costs):
        m.addConstr(gp.quicksum(probabilities[i] * unopt_solution.costs[i, j + 1] for i in range(num_policies))
                    <= env.secondary_cost_bounds[j])

    # Objective: minimise expected cost by choice of probability distribution
    obj = m.addVar(name="objective", lb=min_cost, ub=max_cost)
    m.addConstr(gp.quicksum(probabilities[i] * unopt_solution.costs[i, 0] for i in range(num_policies)) == obj)
    m.setObjective(obj, GRB.MINIMIZE)

    # If user has requested enforcement of any of the constraints that require calculating worst-case-value
    if "tradeoff_wcv" in constraint_params or "bound_wcv" in constraint_params or "bound_ewd" in constraint_params:
        policy_active = m.addVars(range(num_policies),
                                  name="policy_active",
                                  vtype=GRB.BINARY)  # i-th var. is 1 iff i-th probability is above `active' threshold
        activated_costs = m.addVars(range(num_policies),
                                    name="activated_costs",
                                    vtype=GRB.CONTINUOUS,
                                    lb=0.0,
                                    ub=max_cost)  # i-th var. is i-th policy's primary cost if policy is active (0 otherwise)

        for i in range(num_policies):  # Enforcing variable definitions documented above with equivalence constraints
            m.addConstr((policy_active[i] == 1) >> (probabilities[i] >= active_policy_epsilon))
            m.addConstr((policy_active[i] == 0) >> (probabilities[i] <= active_policy_epsilon - 10e-10))
            m.addConstr(activated_costs[i] == policy_active[i] * unopt_solution.costs[i, 0])

        # Worst-case value (highest expected primary cost of an active det. policy)
        worst = m.addVar(name="worst_case_value", lb=min_cost, ub=max_cost)
        m.addConstr(worst == gp.max_(activated_costs))  # Can be recovered as max over the activated_costs vars

        if "bound_wcv" in constraint_params:
            # Upper bounding the worst-case value
            # Bound given by constraint_params["bound_wcv"]
            m.addConstr(worst <= constraint_params["bound_wcv"])
        if "tradeoff_wcv" in constraint_params:
            # Trading off expected value decreases with worst-case value increases
            # Tradeoff rate given by constraint_params["tradeoff_wcv"]
            m.addConstr(unopt_solution.value - obj >= constraint_params["tradeoff_wcv"]
                        * (worst - unopt_solution.worst_case_value) - tradeoff_relaxation)
        if "bound_ewd" in constraint_params:
            # Upper bounding the difference between the expected value and the worst-case value
            # Bound given by constraint_params["bound_ewd"]
            m.addConstr(worst - obj <= constraint_params["bound_ewd"])

    # If user has requested enforcement of the CVaR tradeoff constraint
    if "tradeoff_cvar" in constraint_params:
        m.params.NonConvex = 2  # configuring solver to accept bilinear constraints

        # The following constraints calc. Conditional value at risk of the policy distribution for confidence (1-alpha)
        # Formulation based on method for calculating CVaR over discrete distributions from:
        #   "Conditional value-at-risk: Theory and Application", Kisiala (2015).

        # Auxiliary variables represent transformations of the actual policy costs, used to calc. value-at-risk below
        var_adjusted_costs = m.addVars(range(num_policies),
                                       name="aux_VaR_costs",
                                       vtype=GRB.CONTINUOUS,
                                       lb=min_cost,
                                       ub=max_cost)
        # Auxiliary variables represent transformations of the actual probability values, used to calc. CVaR+ below
        sup_var_probabilities = m.addVars(range(num_policies),
                                          name="sup_var_prob",
                                          lb=0.0,
                                          ub=1.0,
                                          vtype=GRB.CONTINUOUS)
        # cumulative prob's i,e, i-th value is prob. of sampling a det. policy w/ primary cost <= i-th policy's cost)
        cum_probabilities = m.addVars(range(num_policies),
                                      name="cumulative_probabilities",
                                      lb=0.0,
                                      ub=1.0,
                                      vtype=GRB.CONTINUOUS)
        # tying cumulative and regular probability variables
        for i in range(num_policies):
            m.addConstr(cum_probabilities[i] == gp.quicksum([probabilities[j] for j in range(i+1)]))

        # Calculating plain value-at-risk for confidence (1-alpha)
        # This requires finding the min. primary cost for which we are >= alpha likely to sample a policy with cost less than or equal to it

        for i in range(num_policies):
            u = m.addVar(name=f"u_{i}", vtype=GRB.BINARY)  # auxiliary variable
            # Equivalence constraint making a policy's 'var_adjusted_cost' equal max available cost if it fails the above condition
            # Leaves var_adjusted_cost equal to actual cost if it meets the above condition
            m.addConstr((u == 1) >> (cum_probabilities[i] >= alpha))
            m.addConstr((u == 0) >> (cum_probabilities[i] <= (alpha - 10e-10)))
            m.addConstr((u == 1) >> (var_adjusted_costs[i] == unopt_solution.costs[i, mc_cost_index]))
            m.addConstr((u == 0) >> (var_adjusted_costs[i] == max_cost))

        # This lets us define value-at-risk as the minimum over the var_adjusted_cost values
        value_at_risk = m.addVar(name="VaR", lb=min_cost, ub=max_cost)
        m.addConstr(value_at_risk == gp.min_(var_adjusted_costs))

        # Calculating 'CVaR+' for confidence (1-alpha)
        # Formally this is the expected value of the policy distribution, considering ONLY those with strictly greater cost than the value-at-risk

        for i in range(num_policies):
            c = unopt_solution.costs[i, mc_cost_index]
            v = m.addVar(name=f"v_{i}", vtype=GRB.BINARY)  # auxiliary variable
            # Equivalence constraint making a policy's `sup_var_probability' equal to 0 if it fails the above condition (cost <= VaR)
            # Leaves sup_var_probability equal to actual probability if it meets the above condition
            m.addConstr((v == 1) >> ((value_at_risk + 10e-6) <= c))
            m.addConstr((v == 0) >> (value_at_risk >= c))
            m.addConstr((v == 1) >> (sup_var_probabilities[i] == probabilities[i]))
            m.addConstr((v == 0) >> (sup_var_probabilities[i] == 0))

        # Calculating the cumulative probability of sampling any policy that meets the condition
        cum_sup_var_prob = m.addVar(name="cum_sup_var_prob", lb=0.0, ub=1.0)
        m.addConstr(cum_sup_var_prob == sup_var_probabilities.sum())
        norm_factor = m.addVar(name="norm_factor", lb=1.0)  # Finds the necessary normalisation factor for this sub-distribution
        m.addConstr(cum_sup_var_prob * norm_factor == 1.0)

        # Defines `non-normalised' CVaR+ i.e. weighted sum of policy's costs, weighted by non-normalised sub-distribution
        nonnorm_cvar_plus = m.addVar(name="non_normalised_cvar_plus", lb=0.0, ub=max_cost)
        m.addConstr(nonnorm_cvar_plus == gp.quicksum([sup_var_probabilities[i] * unopt_solution.costs[i, mc_cost_index]
                                                      for i in range(num_policies)]))
        # Defines `normalised' (real) CVaR+ by multiplying the non-normalised value and the normalisation factor
        cvar_plus = m.addVar(name="normalised_cvar_plus", lb=min_cost, ub=max_cost)
        m.addConstr(nonnorm_cvar_plus * norm_factor == cvar_plus)

        # Defining lambda, which is a param. for the convex combination below.
        lmd = m.addVar(name="lambda")
        m.addConstr(lmd == (1 - cum_sup_var_prob - alpha) / (1 - alpha))

        # Defining CVaR as a convex combination of VaR and CVaR+
        conditional_value_at_risk = m.addVar(name="CVaR", lb=min_cost, ub=max_cost)
        m.addConstr(conditional_value_at_risk == lmd * value_at_risk + cvar_plus - (lmd * cvar_plus))

        # Trading off expected value decreases with CVaR increases
        # Tradeoff rate given by constraint_params["tradeoff_cvar"]
        m.addConstr(unopt_solution.value - obj >= constraint_params["tradeoff_cvar"]
                    * (conditional_value_at_risk - unopt_solution.cvar) - tradeoff_relaxation)

    m.optimize()

    # Combining the newly optimised distribution and the unoptimised solution to create a new optimised solution
    optimised_solution = unopt_solution
    try:
        optimized_probabilities = np.empty(num_policies)
        for i in range(num_policies):
            optimized_probabilities[i] = probabilities[i].x
        active_policies = optimized_probabilities >= active_policy_epsilon

        optimised_wcv: float
        if "tradeoff_wcv" in constraint_params or "bound_wcv" in constraint_params or "bound_ewd" in constraint_params:
            optimised_wcv = m.getVarByName("worst_case_value").x
        else:
            optimised_wcv = unopt_solution.worst_case_value
        optimised_cvar: float
        if "tradeoff_cvar" in constraint_params:
            optimised_cvar = m.getVarByName("CVaR").x
        else:
            optimised_cvar = unopt_solution.cvar

        optimised_solution = StAnMcCsspSolution(optimized_probabilities[active_policies],
                                                unopt_solution.costs[active_policies, :],
                                                unopt_solution.policies[active_policies],
                                                obj.x,
                                                optimised_wcv,
                                                optimised_cvar)

    except AttributeError:
        pass

    return optimised_solution
