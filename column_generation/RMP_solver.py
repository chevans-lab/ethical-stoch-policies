import gurobipy as gp
from gurobipy import GRB
import numpy as np
from typing import List, Tuple
from column_generation.column_gen_cssp_solution import ColumnGenCSSPSolution
from ethical_cssp_env import EthicalCsspEnv


def solve_rmp(env: EthicalCsspEnv,
              unopt_solution: ColumnGenCSSPSolution,
              enforce_additionals: List[Tuple[bool, float]],
              wellbeing_cost_index=0,
              alpha=0.5):

    """
    The various value parameters which carry over from the previous iteration of the overall loop
    refer specifically to value according to the C-SSPs wellbeing-based cost function (omitted in individual parameter
    definitions for brevity)

    Args:
        policy_costs: 2D array of costs incurred by deterministic policies in RMP
                        (rows are policies, columns are cost functions)
        last_v: Expected cost of the stochastic policy as of the last iteration
        last_cvar: The conditional value-at-risk (confidence 1-alpha) as of the last iteration
        last_worst: Highest expected cost of a deterministic policy in last iteration's stochastic policy
                    with non-zero probability
        cost_bounds: The C-SSP upper bounds on secondary costs (primary cost is to be included in the list at index 0,
                    but is assumed to be None)
        wellbeing_cost_index: The index of the cost function amongst our k functions which is 'wellbeing-based' (MUST BE 0)
        alpha: Defined as (1 - confidence), hyperparameter to conditional value-at-risk
        enforce_additionals: List of bool, float tuples declaring whether to enforce each of the 6 constraints on risk:
                -- 0. If bool, then float will be an upper bound on wellbeing cost of worst det. policy in the mix
                -- 1. If bool, then each iteration must decrease expected wellbeing cost by <float> times the increase
                        in wellbeing cost of worst det. policy in the mix
                -- 2. If bool, then float will be upper bound on diff. between best and worst wellbeing costs of
                        det. policies in the mix
                -- 3. If bool, then each iteration must decrease expected wellbeing cost by <float> times the increase
                        in best/worst wellbeing gap
                -- 4. If bool, then float will be the upper bound on Conditional Value-at-Risk_alpha
                -- 5. If bool, then each iteration must decrease expected wellbeing cost by <float> times the increase
                        in CVaR_alpha of the wellbeing cost

    Returns:

    """

    m = gp.Model("RMP")

    num_policies = unopt_solution.costs.shape[0]

    increasing_cost_order = unopt_solution.costs[:, wellbeing_cost_index].argsort()
    unopt_solution.costs = unopt_solution.costs[increasing_cost_order]
    unopt_solution.policies = unopt_solution.policies[increasing_cost_order]
    max_cost = np.max(unopt_solution.costs[:, wellbeing_cost_index])
    min_cost = np.min(unopt_solution.costs[:, wellbeing_cost_index])

    # Basic RMP constraints and objective
    probabilities = m.addVars(range(num_policies), name="prob", ub=1.0, vtype=GRB.CONTINUOUS)

    # Enforcing normalised probabilities
    m.addConstr(probabilities.sum() == 1.0)
    # Enforcing C-SSP secondary cost bounds are obeyed
    for j in range(len(env.secondary_cost_bounds)):
        m.addConstr(gp.quicksum(probabilities[i] * unopt_solution.costs[i, j + 1] for i in range(num_policies))
                    <= env.secondary_cost_bounds[j])

    # Minimizing expected primary cost
    obj = m.addVar(name="obj", lb=min_cost, ub=max_cost)
    m.addConstr(gp.quicksum(probabilities[i] * unopt_solution.costs[i, 0] for i in range(num_policies)) == obj)
    m.setObjective(obj, GRB.MINIMIZE)


    if enforce_additionals[0][0] or enforce_additionals[1][0] or enforce_additionals[2][0] or enforce_additionals[3][0]:
        active_policy_epsilon = 10e-6
        prob_nonzero = m.addVars(range(num_policies), name="prob_nonzero", vtype=GRB.BINARY)
        activated_costs = m.addVars(range(num_policies), name="activated_costs", vtype=GRB.CONTINUOUS, lb=0, ub=max_cost)
        for i in range(num_policies):
            m.addConstr((prob_nonzero[i] == 1) >> (probabilities[i] >= active_policy_epsilon))
            m.addConstr((prob_nonzero[i] == 0) >> (probabilities[i] <= active_policy_epsilon - 10e-7))
            m.addConstr(activated_costs[i] == prob_nonzero[i] * unopt_solution.costs[i, 0])

        worst = m.addVar(name="worst_activated", lb=0, ub=max_cost)
        m.addConstr(worst == gp.max_(activated_costs))

        if enforce_additionals[0][0]:
            m.addConstr(worst <= enforce_additionals[0][1])
        if enforce_additionals[1][0]:
            m.addConstr(unopt_solution.value - obj >= enforce_additionals[1][1]
                        * (worst - unopt_solution.worst_case_value))
        if enforce_additionals[2][0]:
            m.addConstr(worst - obj <= enforce_additionals[2][1])
        if enforce_additionals[3][0]:
            m.addConstr(unopt_solution.value - obj >= enforce_additionals[3][1]
                        * ((worst - obj) - (unopt_solution.worst_case_value - unopt_solution.value)))

    # Optionally adding CVaR based constraints
    if enforce_additionals[4][0] or enforce_additionals[5][0]:
        m.params.NonConvex = 2

        var_adjusted_costs = m.addVars(range(num_policies), name="aux_VaR_costs", vtype=GRB.CONTINUOUS, lb=min_cost, ub=max_cost)
        sup_var_probabilities = m.addVars(range(num_policies), name="sup_var_prob", ub=1.0, vtype=GRB.CONTINUOUS)
        cum_probabilities = m.addVars(range(num_policies), name="cum_prob", ub=1.0, vtype=GRB.CONTINUOUS)

        for i in range(num_policies):
            m.addConstr(cum_probabilities[i] == gp.quicksum([probabilities[j] for j in range(i+1)]))

        for i in range(num_policies):
            u = m.addVar(name=f"u_{i}", vtype=GRB.BINARY)
            m.addConstr((u == 1) >> (cum_probabilities[i] >= alpha))
            m.addConstr((u == 0) >> (cum_probabilities[i] <= (alpha - 10e-6)))
            m.addConstr((u == 1) >> (var_adjusted_costs[i] == unopt_solution.costs[i, wellbeing_cost_index]))
            m.addConstr((u == 0) >> (var_adjusted_costs[i] == max_cost))

        value_at_risk = m.addVar(name="VaR", lb=min_cost, ub=max_cost)
        m.addConstr(value_at_risk == gp.min_(var_adjusted_costs))

        for i in range(num_policies):
            c = unopt_solution.costs[i, wellbeing_cost_index]
            v = m.addVar(name=f"v_{i}", vtype=GRB.BINARY)
            m.addConstr((v == 1) >> ((value_at_risk + 10e-6) <= c))
            m.addConstr((v == 0) >> (value_at_risk >= c))
            m.addConstr((v == 1) >> (sup_var_probabilities[i] == probabilities[i]))
            m.addConstr((v == 0) >> (sup_var_probabilities[i] == 0))

        cum_sup_var_prob = m.addVar(name="cum_sup_var_prob", ub=1.0)
        m.addConstr(cum_sup_var_prob == sup_var_probabilities.sum())
        norm_factor = m.addVar(name="norm_factor", lb=1.0)
        m.addConstr(cum_sup_var_prob * norm_factor == 1.0)

        nonnorm_cvar_plus = m.addVar(name="non_normalised_cvar_plus", ub=max_cost)
        cvar_plus = m.addVar(name="normalised_cvar_plus", lb=min_cost, ub=max_cost)
        m.addConstr(nonnorm_cvar_plus * norm_factor == cvar_plus)
        m.addConstr(nonnorm_cvar_plus == gp.quicksum([sup_var_probabilities[i] * unopt_solution.costs[i, wellbeing_cost_index]
                                                      for i in range(num_policies)]))

        lmd = m.addVar(name="lambda")
        m.addConstr(lmd == (1 - cum_sup_var_prob - alpha) / (1 - alpha))

        conditional_value_at_risk = m.addVar(name="CVaR", lb=min_cost, ub=max_cost)
        #m.addConstr(value_at_risk == conditional_value_at_risk)
        m.addConstr(conditional_value_at_risk == lmd * value_at_risk + cvar_plus - (lmd * cvar_plus))

        if enforce_additionals[4][0]:
            m.addConstr(conditional_value_at_risk <= enforce_additionals[4][1])
        if enforce_additionals[5][0]:
            m.addConstr(unopt_solution.value - obj >= enforce_additionals[5][1]
                        * (conditional_value_at_risk - unopt_solution.cvar))

    m.optimize()

    try:
        optimized_probabilities = np.empty(num_policies)
        for i in range(num_policies):
            optimized_probabilities[i] = probabilities[i].x
        active_policies = optimized_probabilities > 10e-6

        unopt_solution.probabilities = optimized_probabilities[active_policies]
        unopt_solution.costs = unopt_solution.costs[active_policies, :]
        unopt_solution.policies = unopt_solution.policies[active_policies]
        unopt_solution.value = obj.x

        if enforce_additionals[0][0] or enforce_additionals[1][0] or enforce_additionals[2][0] or enforce_additionals[3][0]:
            unopt_solution.worst_case_value = m.getVarByName("worst_activated").x
        if enforce_additionals[4][0] or enforce_additionals[5][0]:
            unopt_solution.cvar = m.getVarByName("CVaR").x
    except AttributeError:
        m.computeIIS()
        m.write("column_generation/model.ilp")

    return unopt_solution

if __name__ == "__main__":
    x = 1

    # policies = np.array([[10, 10],
    #                     [5, 12],
    #                     [8, 8],
    #                     [10, 12],
    #                     [2, 15],
    #                     [1, 17]])
    # cost_bounds = [None, 13]
    # last_v = 2
    # last_cvar = 5
    # alpha = 0.5

    # solve_rmp(policies, last_v, last_cvar, cost_bounds, 0, alpha)










