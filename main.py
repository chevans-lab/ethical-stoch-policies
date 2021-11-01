from doctor_env import DoctorEnv, DoctorState, DoctorAction, DoctorActionName
import doctor_env
from itertools import chain, product, combinations
from skdecide import DiscreteDistribution
import solver
import gurobi_solver
from matplotlib import pyplot as plt
from column_generation.det_SSP_solver import optimal_deterministic_policy
from column_generation.CG_main import solve_cssp



def main():
    action_list = [DoctorAction(DoctorActionName.B, 600, DiscreteDistribution([(3, 0.25), (5, 0.25), (6, 0.5)])),
                   DoctorAction(DoctorActionName.A, 1000, DiscreteDistribution([(5, 0.25), (6, 0.25), (10, 0.5)])),
                   DoctorAction(DoctorActionName.C, 500, DiscreteDistribution([(0, 0.2), (5, 0.8)])),
                   DoctorAction(DoctorActionName.Discharge, 0, None)]

    action_combinations = [set(c) for c in chain.from_iterable(combinations(action_list, r) for r in range(len(action_list)+1))]
    state_space = [DoctorState(i, *params) for i, params in enumerate(product(list(range(11)), action_combinations))]

    goal_states = [state for state in state_space if doctor_env.terminal_state(state)]

    initial_state = [state for state in state_space if state.pain == 10 and len(state.actions_taken) == 0][0]

    secondary_cost_bounds = [1200]
    wellbeing_costs = [True, False]

    doctor_instance = DoctorEnv(state_space,
                                initial_state,
                                goal_states,
                                action_list,
                                secondary_cost_bounds,
                                len(secondary_cost_bounds),
                                wellbeing_costs)

    policy, flow = solver.find_policy(doctor_instance, enforce_secondaries=True, enforce_ethical=False)

    print("Policy computed")
    for action, prob in policy[initial_state].get_values():
        print(action.name, prob)

    instantiations = []
    pains = []
    costs = []
    for i in range(1000):
        p, c, a_name_list = doctor_instance.simulate_run(policy)
        pains.append(p)
        costs.append(c)
        instantiations.append(set(a_name_list))

    plt.hist(pains)
    plt.title("Pain Rating upon Discharge")
    plt.show()

    solution = solve_cssp(doctor_instance)
    print(solution.costs)
    print(solution.policies)
    print(solution.probabilities)
    print(solution.cvar)


if __name__ == "__main__":
    main()
