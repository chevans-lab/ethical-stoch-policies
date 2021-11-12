from env import medic_env
from matplotlib import pyplot as plt
from stan_MC.CG_main import solve_cssp
import numpy as np
import pandas as pd
import seaborn as sns

def solve_and_plot():

    doctor_instance = medic_env.construct_instance('medic_small')

    solution_iterations = 20
    inner_iterations = 20
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

if __name__ == "__main__":
    solve_and_plot()
