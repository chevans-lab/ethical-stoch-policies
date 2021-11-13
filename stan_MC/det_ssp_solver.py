from env.mc_cssp_env import MorallyConsequentialCsspEnv, ActionName
from stan_MC.stan_mc_cssp_solution import StAnMcCsspSolution

from typing import Dict
import gurobipy as gp
from gurobipy import GRB
import numpy as np

"""
Author: Charles Evans
Email: u6942700@anu.edu.au
This is my own work, and forms part of my artefact contribution for COMP3770, Semester 1, 2021.
"""


def optimal_deterministic_policy(env: MorallyConsequentialCsspEnv):
    """
    Computes and returns the optimal feasible deterministic policy for the C-SSP instance `env'.
    Formulated as the MIP extension of the Dual LP for C-SSPs. Encoding follows the formulation in:
        "Stationary Deterministic Policies for Constrained MDPs with Multiple Rewards, Costs, and Discount Factors", Dolgov & Durfee (2005)

    Solution is trivially morally `acceptable' w.r.t. policy stochasticity (because it has zero policy stochasticity).
    Returned as a special case of a StAnMcCsspSolution, with only a single deterministic policy (the optimal),
    which will be sampled from the StAnMcCsspSolution object with probability 1.0

    Args:
        env: The morally consequential C-SSP instance.

    Returns:
        StAnMcCsspSolution
    """

    m = gp.Model("optimal-det-cssp-policy")

    # State -> (Action -> Flow) mapping
    flow_variables: Dict[int, Dict[ActionName, gp.Var]] = {}
    # State -> in-/out-flow mapping
    out_variables: Dict[int, gp.Var] = {}
    in_variables_transient: Dict[int, gp.Var] = {}
    in_variables_goal: Dict[int, gp.Var] = {}
    # State -> (Action -> binary int.) mapping
    policy_variables: Dict[int, Dict[ActionName, gp.Var]] = {}

    # Creating flow (continuous) and policy (binary) variables for each transient state and applicable action pair
    for s in env.state_space:
        for a in env.applicable_actions(s):
            flow = m.addVar(name=f"x_{s.id}_{a.name}")
            policy = m.addVar(name=f"d_{s.id}_{a.name}", vtype=GRB.BINARY)
            # Storing variables in respective dicts
            if s.id not in flow_variables:
                flow_variables[s.id] = {}
                policy_variables[s.id] = {}
            flow_variables[s.id][a.name] = flow
            policy_variables[s.id][a.name] = policy
            m.addConstr(flow <= policy)  # Forces policy var to be 1 for a state-action pair with nonzero flow

    # Creating, constraining and storing out(s) variables for all transient states s
    for s in env.state_space:
        if not env.terminal_state(s):
            out_s = m.addVar(name=f"out_{s.id}")
            out_variables[s.id] = out_s

            if flow_variables[s.id]:
                # out(s) == sum of outflows for all applicable actions from s
                m.addConstr(out_s == gp.quicksum(list(flow_variables[s.id].values())))
                # Only one state-action pair can have out-flow for the state (deterministic policy)
                m.addConstr(1 >= gp.quicksum(list(policy_variables[s.id].values())))
            else:
                m.addConstr(out_s == 0)

    # Creating, constraining and storing in(s) variables for all transient states s
    for s in env.state_space:
        in_s = m.addVar(name=f"in_{s.id}")
        if env.terminal_state(s):
            in_variables_goal[s.id] = in_s
        else:
            in_variables_transient[s.id] = in_s

        # Finding all transition 'parents' of s, and temp. storing their flow variables and the transition probabilities
        flow_parents = []
        succession_probabilities = []
        for s_ in env.state_space:
            for a in env.applicable_actions(s_):
                transition_probabilities = env.transition_probabilities(s_, a)
                for element, probability in transition_probabilities.get_values():
                    if element.id == s.id:
                        succession_probabilities.append(probability)
                        flow_parents.append(flow_variables[s_.id][a.name])

        # in(s) == transition-probability-weighted sum of flows of transition parents
        if flow_parents:
            m.addConstr(in_s == flow_parents @ np.array(succession_probabilities))
        else:
            m.addConstr(in_s == 0)

    # Enforcing conversation of flow
    for s in env.state_space:
        if not env.terminal_state(s):
            aggregate_flow = 0
            if s.id == env.initial_state.id:
                # out(s) - in(s) = 1 for initial state (0 for all other transient states)
                aggregate_flow += 1
            m.addConstr(out_variables[s.id] - in_variables_transient[s.id] == aggregate_flow)

    # Enforcing a complete flow into goal states
    m.addConstr(gp.quicksum(list(in_variables_goal.values())) == 1)

    # Introducing objective variable (expected primary cost)
    aggregate_costs = []
    obj = m.addVar(name="obj")
    aggregate_costs.append(obj)
    # Constraining to equal the flow-weighted sum of the primary cost of valid state-action pairs
    m.addConstr(obj == gp.quicksum([flow_variables[s.id][a.name] * env.transition_costs(s, a)[0]
                                    for s in env.state_space for a in env.applicable_actions(s)
                                    if not env.terminal_state(s)]))

    # Introducing expected secondary cost variables
    for j in range(1, env.num_secondary_costs + 1):
        exp_cost_j = m.addVar(name="exp_cost_j")
        aggregate_costs.append(exp_cost_j)
        # Constraining to equal the flow-weighted sum of the j-th cost of valid state-action pairs
        m.addConstr(exp_cost_j == gp.quicksum([flow_variables[s.id][a.name] * env.transition_costs(s, a)[j]
                                               for s in env.state_space for a in env.applicable_actions(s)
                                               if not env.terminal_state(s)]))
        # Constraining to not exceed the j-th cost function's in-expectation upper bound
        m.addConstr(exp_cost_j <= env.secondary_cost_bounds[j - 1])

    m.setObjective(obj, GRB.MINIMIZE)
    m.params.LogToConsole = 0
    m.optimize()

    # Creating policy object to be returned (converts variable values to constants)
    return_policy = {}
    for s in env.state_space:
        if not env.terminal_state(s):
            return_policy[s.id] = {}
            for a in env.applicable_actions(s):
                return_policy[s.id][a.name] = policy_variables[s.id][a.name].x

    aggregate_costs = np.array([c.x for c in aggregate_costs])  # in-expectation cost vector of policy

    return StAnMcCsspSolution(probabilities=np.array([1.0]),
                              costs=np.array([aggregate_costs]),
                              policies=np.array([return_policy]),
                              value=aggregate_costs[0],
                              worst_case_value=aggregate_costs[0],  # equals expected value in this special deterministic case
                              cvar=aggregate_costs[0])  # equals expected value in this special deterministic case
