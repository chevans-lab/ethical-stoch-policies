import gurobipy as gp
from gurobipy import GRB
from doctor_env import DoctorEnv
import itertools as it
from typing import Dict
import numpy as np

m = gp.Model("ethical-cssp")

def solve(env: DoctorEnv, enforce_secondaries: bool = True, enforce_ethical: bool = True):
    MAX_Q_VALUES = [10, 2100]
    BIG_I_PENALTY = 10e6
    WEIGHT = 2

    flow_variables: Dict[int, Dict[int, Dict[str, gp.Var]]] = {}
    v_variables: Dict[int, Dict[int, gp.Var]] = {}
    q_variables: Dict[int, Dict[int, Dict[str, gp.Var]]] = {}
    out_variables: Dict[int, Dict[int, float]] = {}
    in_variables_transient: Dict[int, Dict[int, float]] = {}
    in_variables_goal: Dict[int, Dict[int, float]] = {}

    if enforce_ethical:
        d_pi_variables: Dict[int, Dict[str, gp.Var]] = {}
        d_v_variables: Dict[int, Dict[int, gp.Var]] = {}
        big_i_variables: Dict[int, Dict[int, Dict[str, gp.Var]]] = {}


    for i in range(len(env.secondary_cost_bounds)+1):
        v_variables[i] = {}
        q_variables[i] = {}
    if enforce_ethical:
        for i in range(len(env.secondary_cost_bounds) + 1):
            d_v_variables[i] = {}
            if i:
                big_i_variables[i] = {}

    flows_from_s_0 = []
    associated_costs = [[] for _ in range(len(env.secondary_cost_bounds) + 1)]

    # Flow Variables
    print("Flow Variables...")
    for s, s_ in it.product(env.state_space, env.state_space):
        if not env.terminal_state(s):
            for a in env.applicable_actions(s_):
                flow = m.addVar(name=f"x_{s.id}_{s_.id}_{a.name}")
                if s.id not in flow_variables:
                    flow_variables[s.id] = {}
                if s_.id not in flow_variables[s.id]:
                    flow_variables[s.id][s_.id] = {}
                flow_variables[s.id][s_.id][a.name] = flow

                if s.id == env.initial_state.id:
                    flows_from_s_0.append(flow)
                    transition_costs = env.transition_costs(s_, a)
                    for i in range(len(env.secondary_cost_bounds) + 1):
                        associated_costs[i].append(transition_costs[i])

    # IN / OUT variables
    print("I/O Variables...")
    for s, s_ in it.product(env.state_space, env.state_space):
        if not env.terminal_state(s):
            if not env.terminal_state(s_):
                out_s_ = m.addVar(name=f"out_{s.id}_{s_.id}")
                if s.id not in out_variables:
                    out_variables[s.id] = {}
                out_variables[s.id][s_.id] = out_s_

                if flow_variables[s.id][s_.id]:
                    m.addConstr(out_s_ == gp.quicksum(list(flow_variables[s.id][s_.id].values())), name=f"out_{s.id}_{s_.id}_c")
                else:
                    m.addConstr(out_s_ == 0, name=f"out_{s.id}_{s_.id}_c")

            in_s_ = m.addVar(name=f"in_{s.id}_{s_.id}")

            if env.terminal_state(s_):
                if s.id not in in_variables_goal:
                    in_variables_goal[s.id] = {}
                in_variables_goal[s.id][s_.id] = in_s_
            else:
                if s.id not in in_variables_transient:
                    in_variables_transient[s.id] = {}
                in_variables_transient[s.id][s_.id] = in_s_

            flow_parents = []
            succession_probabilities = []
            for s__ in env.state_space:
                for a in env.applicable_actions(s__):
                    transition_probabilities = env.transition_probabilities(s__, a)
                    for element, probability in transition_probabilities.get_values():
                        if element.id == s_.id:
                            succession_probabilities.append(probability)
                            flow_parents.append(flow_variables[s.id][s__.id][a.name])
                            break

            if flow_parents:
                m.addConstr(in_s_ == flow_parents @ np.array(succession_probabilities), name=f"in_{s.id}_{s_.id}_c")
            else:
                m.addConstr(in_s_ == 0, name=f"in_{s.id}_{s_.id}_c")

    # V Variables
    print("V(s) variables...")
    for s in env.state_space:
        v_primary = m.addVar(name=f"v_{s.id}_0")
        v_variables[0][s.id] = v_primary

        for j in range(len(env.secondary_cost_bounds)):
            v_sec = m.addVar(name=f"v_{s.id}_{j+1}")
            v_variables[j+1][s.id] = v_sec

    # Q Variables
    print("Q(s,a) variables...")
    for s in env.state_space:
        for a in env.applicable_actions(s):
            q_primary = m.addVar(name=f"q_{s.id}_{a.name}_0")
            if s.id not in q_variables[0]:
                q_variables[0][s.id] = {}
            q_variables[0][s.id][a.name] = q_primary

            for j in range(len(env.secondary_cost_bounds)):
                q_sec = m.addVar(name=f"q_{s.id}_{a.name}_{j+1}")
                if s.id not in q_variables[j+1]:
                    q_variables[j+1][s.id] = {}
                q_variables[j+1][s.id][a.name] = q_sec

    #CONSTRAINTS

    # Conservation of flow
    print("Enforcing conservation of flow in all sub-problems...")
    for s, s_ in it.product(env.state_space, env.state_space):
        aggregate_flow = 0
        if not env.terminal_state(s) and not env.terminal_state(s_):
            if s.id == s_.id:
                aggregate_flow += 1
            m.addConstr(out_variables[s.id][s_.id] - in_variables_transient[s.id][s_.id] == aggregate_flow, name=f"flow cons_{s.id}_{s_.id}")

    # Linking V to flow
    print("Linking V(s) to the flow variables for all sub-problems...")
    for s in env.state_space:
        if env.terminal_state(s):
            for i in range(len(env.secondary_cost_bounds) + 1):
                m.addConstr(v_variables[i][s.id] == 0, name=f"v_{s.id}_{i}_c")
        else:
            successors_flows = []
            successor_flow_costs = [[] for _ in range(len(env.secondary_cost_bounds) + 1)]
            for s_ in env.state_space:
                for a in env.applicable_actions(s_):
                    successors_flows.append(flow_variables[s.id][s_.id][a.name])
                    costs = env.transition_costs(s_, a)
                    for i in range(len(env.secondary_cost_bounds) + 1):
                        successor_flow_costs[i].append(costs[i])

            for i in range(len(env.secondary_cost_bounds) + 1):
                m.addConstr(v_variables[i][s.id] == successors_flows @ np.array(successor_flow_costs[i]),  name=f"v_{s.id}_{i}_c")

    # Linking Q to V
    print("Linking Q(s,a) to V(s')...")
    for s in env.state_space:
        for a in env.applicable_actions(s):
            successor_values = [[] for _ in range(len(env.secondary_cost_bounds) + 1)]

            transition_probabilities = env.transition_probabilities(s, a)
            for element, _ in transition_probabilities.get_values():
                for i in range(len(env.secondary_cost_bounds) + 1):
                    successor_values[i].append(v_variables[i][element.id])

            probabilities = np.array([prob for _, prob in transition_probabilities.get_values()])

            transition_costs = env.transition_costs(s, a)
            for i in range(len(env.secondary_cost_bounds) + 1):
                m.addConstr(q_variables[i][s.id][a.name] == transition_costs[i] + successor_values[i] @ probabilities,  name=f"q_{s.id}_{a.name}_{i}_c")

    # Linking the flow variables across the |S| sub-problems
    print("Linking flows in sub-problems...")
    for s in env.state_space:
        if env.initial_state.id != s.id and not env.terminal_state(s):
            for s_ in env.state_space:
                for a in env.applicable_actions(s_):
                    if a.name not in flow_variables[s.id][s_.id] or a.name not in flow_variables[env.initial_state.id][s_.id]:
                        assert a.name not in flow_variables[s.id][s_.id] and a.name not in flow_variables[env.initial_state.id][s_.id]
                    else:
                        m.addConstr(flow_variables[s.id][s_.id][a.name] * out_variables[env.initial_state.id][s_.id] ==
                                    flow_variables[env.initial_state.id][s_.id][a.name] * out_variables[s.id][s_.id],
                                    name=f"tie_flow_{s_.id}_{a.name}_with_s0_for_{s.id}")

    # Complete flow into goal states for all sub-problems
    print("Enforcing Complete Flow into goal states...")
    for s in env.state_space:
        if not env.terminal_state(s):
            m.addConstr(gp.quicksum(list(in_variables_goal[s.id].values())) == 1, name=f"complete_from_{s.id}")

    # Secondary cost bounds
    if enforce_secondaries:
        print("Enforcing secondary cost bounds...")
        for j in range(len(env.secondary_cost_bounds)):
            m.addConstr(flows_from_s_0 @ np.array(associated_costs[j + 1]) <= env.secondary_cost_bounds[j], name=f"bound_{j}")

    if enforce_ethical:
        # Deterministic policy variables
        print("Det. Policy variables...")
        for s in env.state_space:
            if not env.terminal_state(s):
                d_pi_s = []
                for a in env.applicable_actions(s):
                    d_pi_s_a = m.addVar(name=f"Det_pi_{s.id}_{a.name}", vtype=GRB.BINARY)
                    d_pi_s.append(d_pi_s_a)
                    if s.id not in d_pi_variables:
                        d_pi_variables[s.id] = {}
                    d_pi_variables[s.id][a.name] = d_pi_s_a
                m.addConstr(gp.quicksum(d_pi_s) == 1, name=f"det_pi_valid_{s.id}")

        # 'Big-I' variables
        for s in env.state_space:
            for a in env.applicable_actions(s):
                for j in range(1, len(env.secondary_cost_bounds) + 1):
                    bigi_j_s_a = m.addVar(name=f"big_I_{s.id}_{a.name}_{j}")
                    if s.id not in big_i_variables[j]:
                        big_i_variables[j][s.id] = {}
                    big_i_variables[j][s.id][a.name] = bigi_j_s_a

        # Deterministic value variables
        for s in env.state_space:
            policy_flows = []
            q_values = [[] for _ in range(len(env.secondary_cost_bounds) + 1)]
            pi_q_products = [[] for _ in range(len(env.secondary_cost_bounds) + 1)]

            big_i_values = [[] for _ in range(len(env.secondary_cost_bounds))]

            if env.terminal_state(s):
                for i in range(len(env.secondary_cost_bounds) + 1):
                    d_s_i = m.addVar(name=f"det_v_{s.id}_{i}")
                    m.addConstr(d_s_i == 0)
            else:
                applicable = env.applicable_actions(s)
                for a in applicable:
                    policy_flows.append(d_pi_variables[s.id][a.name])
                    for i in range(len(env.secondary_cost_bounds) + 1):
                        q_values[i].append(q_variables[i][s.id][a.name])
                    for j in range(1, len(env.secondary_cost_bounds) + 1):
                        big_i_values[j - 1].append(big_i_variables[j][s.id][a.name])

                for i in range(len(env.secondary_cost_bounds) + 1):
                    for a_index in range(len(applicable)):
                        a = applicable[a_index]
                        product_var = m.addVar(name=f"pi_q_prod_{s.id}_{a.name}_{i}")
                        m.addConstr(product_var <= policy_flows[a_index] * MAX_Q_VALUES[i], name=f"A_{s.id}_{a.name}_{i}")
                        m.addConstr(product_var <= q_values[i][a_index], name=f"B_{s.id}_{a.name}_{i}")
                        m.addConstr(product_var >= q_values[i][a_index] - (1 - policy_flows[a_index]) * MAX_Q_VALUES[i], name=f"C_{s.id}_{a.name}_{i}")
                        pi_q_products[i].append(product_var)

                for i in range(len(env.secondary_cost_bounds) + 1):
                    d_s_i = m.addVar(name=f"det_v_{s.id}_{i}")
                    m.addConstr(d_s_i == gp.quicksum(pi_q_products[i]))
                    d_v_variables[i][s.id] = d_s_i

                    if not i:
                        penalised_q = m.addVars(len(applicable), name=f"pen_q_{s.id}")
                        for a_index in range(len(penalised_q)):
                            m.addConstr(penalised_q[a_index] == q_values[0][a_index] + gp.quicksum([p[a_index] for p in big_i_values]))

                        m.addConstr(d_s_i == gp.min_(penalised_q))

        for j in range(1, len(env.secondary_cost_bounds) + 1):
            for s in env.state_space:
                v_s0_j = v_variables[j][env.initial_state.id]
                for a in env.applicable_actions(s):
                    bound_diff = m.addVar(name=f"bound_diff_{s.id}_{a.name}_{j}", lb=-1 * float('inf'))
                    m.addConstr(bound_diff == BIG_I_PENALTY * (v_s0_j
                                                               + flow_variables[env.initial_state.id][s.id][a.name] * q_variables[j][s.id][a.name]
                                                               - flow_variables[env.initial_state.id][s.id][a.name] * v_variables[j][s.id]
                                                               - env.secondary_cost_bounds[j - 1]), name=f"big_I_diff_calc_{s.id}_{a.name}_{j}")
                    m.addConstr(big_i_variables[j][s.id][a.name] == gp.max_(0, bound_diff), name=f"big_I_max_{s.id}_{a.name}_{j}")

        for i in range(1, len(env.secondary_cost_bounds) + 1):
            if env.wellbeing_costs[i]:
                for s in env.state_space:
                    if not env.terminal_state(s):
                        m.addConstr(d_v_variables[i][s.id] - v_variables[i][s.id] >=
                                    WEIGHT * (gp.max_(list(q_variables[i][s.id].values())) - d_v_variables[i][s.id]),
                                    name=f"ethical_{s.id}_{i}")

    obj = flows_from_s_0 @ np.array(associated_costs[0])
    m.setObjective(obj, GRB.MINIMIZE)

    print("Solving...")
    m.params.NonConvex = 2
    m.params.Presolve = 0
    m.optimize()
    #m.computeIIS()
    #m.write("model.ilp")
    print("------------------")
    print("Variable values...")
    for var in m.getVars():
        if var.varName.startswith(f"x_{env.initial_state.id}") and var.x > 10e-6:
            print('%s %g' % (var.varName, var.x))
        if var.varName.startswith(f"v_{env.initial_state.id}"):
            print('%s %g' % (var.varName, var.x))
        if var.varName.startswith(f"q_{env.initial_state.id}_ActionName.A") or var.varName.startswith(f"q_{env.initial_state.id}_ActionName.B"):
            print('%s %g' % (var.varName, var.x))
        if (var.varName.startswith("x_65_65") or var.varName.startswith("x_66_66")) and var.x > 10e-6:
            print('%s %g' % (var.varName, var.x))
        #if var.varName.startswith("big_I"):
        #    print('%s %g' % (var.varName, var.x))

    print()
    print()
    for s in env.state_space:
        print(f"{s.id}: Pain={s.pain} and Actions taken={[a.name for a in s.actions_taken]}")

    return None


































