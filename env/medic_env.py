from typing import Set, List, Tuple, Dict
from env.ethical_cssp_env import MorallyConsequentialCsspEnv, State, Action, ActionName
from skdecide import DiscreteDistribution
from itertools import chain, product, combinations



class MedicActionName(ActionName):
    A = 1
    B = 2
    C = 3
    Discharge = 4


class MedicAction(Action):
    name: MedicActionName
    monetary_cost: int
    effect: DiscreteDistribution[int]

    def __init__(self, name, monetary_cost, effect):
        super().__init__(name)
        self.monetary_cost = monetary_cost
        self.effect = effect


class MedicState(State):
    pain: int
    actions_taken: Set[MedicAction]

    def __init__(self, identification, pain, actions_taken):
        super().__init__(identification)
        self.pain = pain
        self.actions_taken = actions_taken


class MedicEnv(MorallyConsequentialCsspEnv):

    state_space: List[MedicState]
    initial_state: MedicState
    goal_states: List[MedicState]
    action_space: List[MedicAction]

    def __init__(self,
                 state_space,
                 initial_state,
                 goal_states,
                 action_space,
                 secondary_cost_bounds,
                 num_secondary_costs,
                 wellbeing_costs):
        super().__init__(state_space,
                         initial_state,
                         goal_states,
                         action_space,
                         secondary_cost_bounds,
                         num_secondary_costs,
                         wellbeing_costs)

    def applicable_actions(self, s: MedicState) -> List[MedicAction]:
        if not [a for a in s.actions_taken if a.name == MedicActionName.Discharge]:
            return [a for a in self.action_space if a not in s.actions_taken and a.name != "terminal"]
        else:
            return []

    def transition_probabilities(self, s: MedicState, a: MedicAction) -> DiscreteDistribution[MedicState]:
        actions_taken = {a for a in s.actions_taken}
        actions_taken.add(a)
        if a in self.applicable_actions(s):
            if a.name == MedicActionName.Discharge:
                successor = [state for state in self.state_space if state.pain == s.pain and state.actions_taken == actions_taken][0]
                return DiscreteDistribution([(successor, 1.0)])
            else:
                pain_distribution = DiscreteDistribution([(max(s.pain - e, 0), prob) for e, prob in a.effect.get_values()])
                successors = []
                for pain, prob in pain_distribution.get_values():
                    successor = [state for state in self.state_space if state.pain == pain and state.actions_taken == actions_taken][0]
                    successors.append((successor, prob))
                return DiscreteDistribution(successors)

    def simulate_run(self, policy: Dict[MedicState, DiscreteDistribution[MedicAction]]) -> Tuple[float, List[float], List[MedicActionName]]:
        s: MedicState = self.initial_state
        actions = []
        while not self.terminal_state(s):
            a = policy[s].sample()
            s = self.transition_probabilities(s, a).sample()
            actions.append(a.name)
        return float(s.pain), [sum([a.monetary_cost for a in s.actions_taken])], actions

    def transition_costs(self, s: MedicState, a: MedicAction) -> List[float]:
        if a.name == MedicActionName.Discharge:
            return [float(s.pain), float(0)]
        else:
            return [float(0.001), float(a.monetary_cost)]

    def terminal_state(self, s: MedicState) -> bool:
        if [a for a in s.actions_taken if a.name == MedicActionName.Discharge]:
            return True
        else:
            return False


def terminal_state(s: MedicState) -> bool:
    if [a for a in s.actions_taken if a.name == MedicActionName.Discharge]:
        return True
    else:
        return False

def construct_instance(name: str) -> MedicEnv:
    if name == 'medic_small':

        action_list = [MedicAction(MedicActionName.B, 600, DiscreteDistribution([(3, 0.25), (5, 0.25), (6, 0.5)])),
                       MedicAction(MedicActionName.A, 1000, DiscreteDistribution([(5, 0.25), (6, 0.25), (10, 0.5)])),
                       MedicAction(MedicActionName.C, 500, DiscreteDistribution([(0, 0.2), (5, 0.8)])),
                       MedicAction(MedicActionName.Discharge, 0, None)]

        action_combinations = [set(c) for c in
                               chain.from_iterable(combinations(action_list, r) for r in range(len(action_list) + 1))]
        state_space = [MedicState(i, *params) for i, params in enumerate(product(list(range(11)), action_combinations))]

        goal_states = [state for state in state_space if terminal_state(state)]

        initial_state = [state for state in state_space if state.pain == 10 and len(state.actions_taken) == 0][0]

        secondary_cost_bounds = [1200]
        wellbeing_costs = [True, False]

        return MedicEnv(state_space,
                        initial_state,
                        goal_states,
                        action_list,
                        secondary_cost_bounds,
                        len(secondary_cost_bounds),
                        wellbeing_costs)