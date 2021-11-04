from doctor_env import DoctorEnv, DoctorState, DoctorAction, DoctorActionName
import doctor_env
from itertools import chain, product, combinations
from skdecide import DiscreteDistribution
from matplotlib import pyplot as plt
from column_generation.CG_main import solve_cssp
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.timeseries



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

    solution_iterations = 200
    inner_iterations = 100
    iteration_index = np.arange(solution_iterations * inner_iterations) % inner_iterations
    iterations_values = np.empty(0)
    worst_values = np.empty(0)
    cvar_values = np.empty(0)

    for i in range(solution_iterations):
        solution, plot_data = solve_cssp(doctor_instance, n_iterations=inner_iterations)
        iterations_values = np.concatenate((iterations_values, plot_data["Value"]))
        if "Worst" in plot_data:
            worst_values = np.concatenate((worst_values, plot_data["Worst"]))
        if "CVaR" in plot_data:
            cvar_values = np.concatenate((cvar_values, plot_data["CVaR"]))
        print(solution.costs)
        print(solution.policies)
        print(solution.probabilities)

    print(cvar_values)

    df = pd.DataFrame({'Iteration': iteration_index, 'Value': iterations_values})

    sns.lineplot(x='Iteration', y='Value', data=df, label='Expected Value')
    if worst_values.size > 0:
        df['Worst'] = worst_values
        sns.lineplot(x='Iteration', y='Worst', data=df, label='Worst-Case Value')
    if cvar_values.size > 0:
        df['CVaR'] = cvar_values
        sns.lineplot(x='Iteration', y='CVaR', data=df, label='Conditional Value at Risk')

    plt.title("Expected vs. Worst-Case Value in Iterated Policy Improvement")
    plt.ylim([0.5, 4.0])
    plt.show()




    #plt.plot(plot_data["Value"], color="red", label="expected policy value")
    #if "Worst" in plot_data:
    #    plt.plot(plot_data["Worst"], color="blue", label="worst case policy value")
    #if "CVaR" in plot_data:
    #    plt.plot(plot_data["CVaR"], color="green", label="conditional value at risk")
    #plt.legend()
    #plt.show()


if __name__ == "__main__":
    main()
