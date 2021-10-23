from ethical_cssp_env import EthicalCsspEnv, State, Action
from typing import List, Dict, Tuple
import gurobipy as gp
from gurobipy import GRB
import numpy as np


def reduced_cost_ssp(env: EthicalCsspEnv,
                     dual_cost: List[float],
                     banned_policies: List[List[Tuple[int, str]]],
                     banned_flows: List[Tuple[int, str]]):

    m = gp.Model("rc-ssp")

    flow_variables: Dict[int, Dict[str, gp.Var]] = {}
    reduced_costs: Dict[int, Dict[str, float]] = {}
    out_variables: Dict[int, gp.Var] = {}
    in_variables_transient: Dict[int, gp.Var] = {}
    in_variables_goal: Dict[int, gp.Var] = {}

    # Flow Variables and Reduced Costs
    for s in env.state_space:
        for a in env.applicable_actions(s):
            flow = m.addVar(name=f"x_{s.id}_{a.name}")
            if s.id not in flow_variables:
                flow_variables[s.id] = {}
            flow_variables[s.id][a.name] = flow

            cost = env.transition_costs(s, a)
            reduced_cost = cost[0]
            for j in range(env.num_secondary_costs):
                reduced_cost -= dual_cost[j] * cost[j + 1]

            if s.id not in reduced_costs:
                reduced_costs[s.id] = {}
            reduced_costs[s.id][a.name] = reduced_cost

    # Out variables
    for s in env.state_space:
        if not env.terminal_state(s):
            out_s = m.addVar(name=f"out_{s.id}")
            out_variables[s.id] = out_s

            if flow_variables[s.id]:
                m.addConstr(out_s == gp.quicksum(list(flow_variables[s.id].values())), name=f"out_{s.id}_c")
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
    print("Enforcing conservation of flow...")
    for s in env.state_space:
        if not env.terminal_state(s):
            aggregate_flow = 0
            if s.id == env.initial_state.id:
                aggregate_flow += 1
            m.addConstr(out_variables[s.id] - in_variables_transient[s.id] == aggregate_flow, name=f"flow_cons_{s.id}")

    # Complete flow into goal states
    print("Enforcing Complete Flow into goal states...")
    m.addConstr(gp.quicksum(list(in_variables_goal.values())) == 1, name=f"complete_flow_to_goal")

    # Enforcing whole-policy bans:
    for policy in banned_policies:
        flow_outside_policy = [flow_variables[s.id][a.name] for s in env.state_space for a in env.applicable_actions(s)
                               if (s.id in flow_variables and (s.id, a.name) not in policy)]
        m.addConstr(gp.quicksum(flow_outside_policy) > 10e-4)

    # Enforcing specific state-action bans
    for state_id, action_name in banned_flows:
        if state_id in flow_variables and action_name in flow_variables[state_id]:
            m.addConstr(flow_variables[state_id][action_name] == 0)

    obj = [flow_variables[s.id][a.name] * reduced_costs[s.id][a.name] for s in env.state_space for a in env.applicable_actions(s) if s.id in flow_variables]
    m.setObjective(gp.quicksum(obj), GRB.MINIMIZE)

    print("Solving...")
    m.write("model.lp")
    m.optimize()

    for var in m.getVars():
        if var.varName.startswith(f"x_") and var.x > 10e-6:
            print('%s %g' % (var.varName, var.x))

    return None




