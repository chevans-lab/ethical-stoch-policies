from env.mc_cssp_env import MorallyConsequentialCsspEnv
from stan_MC.stan_mc_cssp_solution import StAnMcCsspSolution

from typing import Dict
import gurobipy as gp
from gurobipy import GRB
import numpy as np


def optimal_deterministic_policy(env: MorallyConsequentialCsspEnv):

    m = gp.Model("optimal-det-cssp-policy")

    flow_variables: Dict[int, Dict[str, gp.Var]] = {}
    out_variables: Dict[int, gp.Var] = {}
    in_variables_transient: Dict[int, gp.Var] = {}
    in_variables_goal: Dict[int, gp.Var] = {}
    policy_variables: Dict[int, Dict[str, gp.Var]] = {}

    # Flow Variables
    for s in env.state_space:
        for a in env.applicable_actions(s):
            flow = m.addVar(name=f"x_{s.id}_{a.name}")
            policy = m.addVar(name=f"d_{s.id}_{a.name}", vtype=GRB.BINARY)
            if s.id not in flow_variables:
                flow_variables[s.id] = {}
                policy_variables[s.id] = {}
            flow_variables[s.id][a.name] = flow
            policy_variables[s.id][a.name] = policy
            m.addConstr(flow <= policy)

    # Out variables
    for s in env.state_space:
        if not env.terminal_state(s):
            out_s = m.addVar(name=f"out_{s.id}")
            out_variables[s.id] = out_s

            if flow_variables[s.id]:
                m.addConstr(out_s == gp.quicksum(list(flow_variables[s.id].values())), name=f"out_{s.id}_c")
                m.addConstr(1 >= gp.quicksum(list(policy_variables[s.id].values())), name=f"pol_out_{s.id}_c")
            else:
                m.addConstr(out_s == 0, name=f"out_{s.id}_c")

    # In variables
    for s in env.state_space:
        in_s = m.addVar(name=f"in_{s.id}")
        if env.terminal_state(s):
            in_variables_goal[s.id] = in_s
        else:
            in_variables_transient[s.id] = in_s

        flow_parents = []
        succession_probabilities = []
        for s_ in env.state_space:
            for a in env.applicable_actions(s_):
                transition_probabilities = env.transition_probabilities(s_, a)
                for element, probability in transition_probabilities.get_values():
                    if element.id == s.id:
                        succession_probabilities.append(probability)
                        flow_parents.append(flow_variables[s_.id][a.name])

        if flow_parents:
            m.addConstr(in_s == flow_parents @ np.array(succession_probabilities), name=f"in_{s.id}_c")
        else:
            m.addConstr(in_s == 0, name=f"in_{s.id}_c")

    # Conservation of flow
    for s in env.state_space:
        if not env.terminal_state(s):
            aggregate_flow = 0
            if s.id == env.initial_state.id:
                aggregate_flow += 1
            m.addConstr(out_variables[s.id] - in_variables_transient[s.id] == aggregate_flow, name=f"flow_cons_{s.id}")

    # Complete flow into goal states
    m.addConstr(gp.quicksum(list(in_variables_goal.values())) == 1, name=f"complete_flow_to_goal")

    aggregate_costs = []
    obj = m.addVar(name="obj")
    aggregate_costs.append(obj)
    m.addConstr(obj == gp.quicksum([flow_variables[s.id][a.name] * env.transition_costs(s, a)[0]
                                    for s in env.state_space for a in env.applicable_actions(s)
                                    if not env.terminal_state(s)]))

    # Enforcing secondary cost constraints
    for j in range(1, env.num_secondary_costs + 1):
        exp_cost_j = m.addVar(name="exp_cost_j")
        aggregate_costs.append(exp_cost_j)
        m.addConstr(exp_cost_j == gp.quicksum([flow_variables[s.id][a.name] * env.transition_costs(s, a)[j]
                                               for s in env.state_space for a in env.applicable_actions(s)
                                               if not env.terminal_state(s)]))
        m.addConstr(exp_cost_j <= env.secondary_cost_bounds[j - 1])

    m.setObjective(obj, GRB.MINIMIZE)

    m.params.LogToConsole = 0
    m.optimize()

    return_policy = {}
    for s in env.state_space:
        if not env.terminal_state(s):
            return_policy[s.id] = {}
            for a in env.applicable_actions(s):
                return_policy[s.id][a.name] = policy_variables[s.id][a.name].x

    aggregate_costs = np.array([c.x for c in aggregate_costs])

    return StAnMcCsspSolution(probabilities=np.array([1.0]),
                              costs=np.array([aggregate_costs]),
                              policies=np.array([return_policy]),
                              value=aggregate_costs[0],
                              worst_case_value=aggregate_costs[0],
                              cvar=aggregate_costs[0])
