from ethical_cssp_env import EthicalCsspEnv, State, Action
from typing import List, Tuple, Dict
from random import randrange
import numpy as np


def random_walk(env: EthicalCsspEnv, n_policies: int):

    policies = []
    cost_vectors = []

    while n_policies > 0:
        flow: Dict[int, Dict[str, float]] = {}
        policy: Dict[int, Dict[str, float]] = {}
        open_states: List[Tuple[State, float]] = [(env.initial_state, 1)]
        established_actions: Dict[int, Action] = {}

        while open_states:
            s, in_flow = open_states.pop(0)
            a: Action
            if s.id in established_actions:
                a = established_actions[s.id]
                flow[s.id][a.name] += in_flow
            else:
                flow[s.id] = {}
                policy[s.id] = {}
                applicable_actions = env.applicable_actions(s)
                for action in env.applicable_actions(s):
                    policy[s.id][action.name] = 0
                a = applicable_actions[randrange(len(applicable_actions))]
                flow[s.id][a.name] = in_flow
                policy[s.id][a.name] = 1
                established_actions[s.id] = a

            for s_, prob in env.transition_probabilities(s, a).get_values():
                if not env.terminal_state(s_):
                    open_states.append((s_, prob * in_flow))

        for s in env.state_space:
            for a in env.applicable_actions(s):
                if s.id not in flow:
                    flow[s.id] = {}
                if a.name not in flow[s.id]:
                    flow[s.id][a.name] = 0

        costs = np.array([flow[s.id][a.name] * np.array(env.transition_costs(s, a))
                         for s in env.state_space for a in env.applicable_actions(s) if s.id in flow])
        summated_costs = np.sum(costs, axis=0)

        policies.append(policy)
        cost_vectors.append(summated_costs)
        n_policies -= 1

    return np.array(cost_vectors), np.array(policies)








