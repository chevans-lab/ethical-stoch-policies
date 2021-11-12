from env import medic_env
from stan_MC.stan_main import solve_cssp

import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from typing import Dict

"""
Author: Charles Evans
Email: u6942700@anu.edu.au
This is my own work, and forms part of my artefact contribution for COMP3770, Semester 1, 2021.
"""


def solve_and_plot(instance_name: str, constraint_params: Dict[str, float], iterations: int, sample_size: int, repetitions: int):
    medic_instance = medic_env.construct_instance(instance_name)

    num_datapoints = repetitions * iterations
    iteration_index = np.arange(num_datapoints) % iterations
    expected_value_data = np.empty(num_datapoints)
    wcv_data = np.empty(num_datapoints)
    ewd_data = np.empty(num_datapoints)
    cvar_data = np.empty(num_datapoints)

    for i in range(repetitions):
        solution, plot_data = solve_cssp(medic_instance,
                                         constraint_params=constraint_params,
                                         iterations=iterations,
                                         sample_size=sample_size,
                                         store_plot_data=True)

        expected_value_data[np.arange(i * iterations, (i + 1) * iterations)] = plot_data["Value"]
        if "Worst" in plot_data:
            wcv_data[np.arange(i * iterations, (i + 1) * iterations)] = plot_data["Worst"]
        if "Expected_Worst_Diff" in plot_data:
            ewd_data[np.arange(i * iterations, (i + 1) * iterations)] = plot_data["Expected_Worst_Diff"]
        if "CVaR" in plot_data:
            cvar_data[np.arange(i * iterations, (i + 1) * iterations)] = plot_data["CVaR"]

    df = pd.DataFrame({'Iteration': iteration_index, 'Expected Value': expected_value_data})

    sns.lineplot(x='Iteration', y='Expected Value', data=df, label='Expected Value')
    if "tradeoff_wcv" in constraint_params or "bound_wcv" in constraint_params:
        df['Worst'] = wcv_data
        sns.lineplot(x='Iteration', y='Worst', data=df, label='Worst-Case Value')
    if "bound_ewd" in constraint_params:
        df['Expected_Worst_Diff'] = ewd_data
        sns.lineplot(x='Iteration', y='Expected_Worst_Diff', data=df, label='Expected-Worst Value Difference')
    if "tradeoff_cvar" in constraint_params:
        df['CVaR'] = cvar_data
        sns.lineplot(x='Iteration', y='CVaR', data=df, label='Conditional Value at Risk')

    plt.title("Evolution of Expected and Disadvantaged Policy Value")
    plt.show()


def positive_int_cast(argument, arg_name):
    value: int
    v_err = False
    try:
        value = int(argument)
        if value <= 0:
            v_err = True
        else:
            return value
    except ValueError:
        v_err = True
    if v_err:
        raise ValueError(f"If provided, the {arg_name} parameter must be a positive integer")


def float_cast(argument, arg_name):
    value: float
    try:
        value = float(argument)
    except ValueError:
        raise ValueError(f"If provided, the {arg_name} parameter must be numeric")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bw", "--bound_wcv", dest="bound_wcv", metavar="WCV_BOUND",
                        help="Upper bounds the worst-case expected value of the policy. Defaults to no bound.")
    parser.add_argument("-be", "--bound_ewd", dest="bound_ewd", metavar="EWD_BOUND",
                        help="Upper bounds the difference between the expected and worst-case expected value of the policy. Defaults to no bound.")
    parser.add_argument("-tw", "--tradeoff_wcv", dest="tradeoff_wcv", metavar="WCV_TRADEOFF_RATE",
                        help="Configures the worst-case value to expected value tradeoff rate when solving for the policy. Defaults to no rate (constraint not enforced).")
    parser.add_argument("-tc", "--tradeoff_cvar", dest="tradeoff_cvar", metavar="CVAR_TRADEOFF_RATE",
                        help="Configures the conditional value at risk (with preconfigured alpha) to expected value tradeoff rate when solving for the policy. Defaults to no rate (constraint not enforced).")

    parser.add_argument("-i", "--instance", dest="instance", metavar="INSTANCE",
                        help="The morally consequential C-SSP instance to be solved. Allowable values are 'medic_small' (default).")
    parser.add_argument("-t", "--iterations", dest="iterations", metavar="ITERATIONS",
                        help="No. policy improvement iterations that StAn-MC will perform. Default value of 50.")
    parser.add_argument("-s", "--sample_size", dest="sample_size", metavar="SAMPLE_SIZE",
                        help="Configures the batch size of deterministic policies randomly sampled by StAn-MC at each timestep. Default value of 20.")
    parser.add_argument("-r", "--repetitions", dest="repetitions", metavar="REPETITIONS",
                        help="No. times the instance will be solved with StAn-MC (the population of solves allows plotting a confidence interval in the results). Default value of 20.")

    args = parser.parse_args()

    # Parsing instance parameter
    instance_name = args.instance
    if instance_name is None:
        instance_name = "medic_small"

    # Parsing constraint parameters
    constraint_params = {}
    bound_wcv = args.bound_wcv
    if bound_wcv is not None:
        bound_wcv = float_cast(bound_wcv, "bound_wcv")
        constraint_params["bound_wcv"] = bound_wcv
    bound_ewd = args.bound_ewd
    if bound_ewd is not None:
        bound_ewd = float_cast(bound_ewd, "bound_ewd")
        constraint_params["bound_ewd"] = bound_ewd
    tradeoff_wcv = args.tradeoff_wcv
    if tradeoff_wcv is not None:
        tradeoff_wcv = float_cast(tradeoff_wcv, "tradeoff_wcv")
        constraint_params["tradeoff_wcv"] = tradeoff_wcv
    tradeoff_cvar = args.tradeoff_cvar
    if tradeoff_cvar is not None:
        tradeoff_cvar = float_cast(tradeoff_cvar, "tradeoff_cvar")
        constraint_params["tradeoff_cvar"] = tradeoff_cvar

    # Parsing solver and plotting parameters
    iterations = args.iterations
    if iterations is None:
        iterations = 50
    else:
        iterations = positive_int_cast(iterations, "iterations")
    sample_size = args.sample_size
    if sample_size is None:
        sample_size = 20
    else:
        sample_size = positive_int_cast(sample_size, "sample_size")
    repetitions = args.repetitions
    if repetitions is None:
        repetitions = 20
    else:
        repetitions = positive_int_cast(repetitions, "repetitions")

    solve_and_plot(instance_name,
                   constraint_params,
                   iterations,
                   sample_size,
                   repetitions)
