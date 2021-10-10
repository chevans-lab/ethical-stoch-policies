from typing import Set, List, Tuple, Dict
from enum import Enum

from skdecide import DiscreteDistribution


class ActionName(Enum):
    A = 1
    B = 2
    C = 3
    D = 4
    Discharge = 5


class Action:
    name: ActionName
    monetary_cost: int
    effect: DiscreteDistribution[int]

    def __init__(self, name, monetary_cost, effect):
        self.name = name
        self.monetary_cost = monetary_cost
        self.effect = effect


class State:
    pain: int
    actions_taken: Set[Action]
    id: int

    def __init__(self, pain, actions_taken, id):
        self.pain = pain
        self.actions_taken = actions_taken
        self.id = id


class DoctorEnv:
    MAX_PAIN = 10

    state_space: List[State]
    initial_state: State
    goal_states: Set[State]
    action_space: List[Action]
    secondary_cost_bounds = List[float]
    wellbeing_costs = List[bool]

    def __init__(self, state_space, initial_state, goal_states, action_space, secondary_cost_bounds, wellbeing_costs):
        self.state_space = state_space
        self.initial_state = initial_state
        self.goal_states = goal_states
        self.action_space = action_space
        self.secondary_cost_bounds = secondary_cost_bounds
        self.wellbeing_costs = wellbeing_costs

    def applicable_actions(self, s: State) -> List[Action]:
        if not [a for a in s.actions_taken if a.name == ActionName.Discharge]:
            return [a for a in self.action_space if a not in s.actions_taken and a.name != "terminal"]
        else:
            return []

    def transition_probabilities(self, s: State, a: Action) -> DiscreteDistribution[State]:
        actions_taken = {a for a in s.actions_taken}
        actions_taken.add(a)
        if a in self.applicable_actions(s):
            if a.name == ActionName.Discharge:
                successor = [state for state in self.state_space if state.pain == s.pain and state.actions_taken == actions_taken][0]
                return DiscreteDistribution([(successor, 1.0)])
            else:
                pain_distribution = DiscreteDistribution([(max(s.pain - e, 0), prob) for e, prob in a.effect.get_values()])
                successors = []
                for pain, prob in pain_distribution.get_values():
                    successor = [state for state in self.state_space if state.pain == pain and state.actions_taken == actions_taken][0]
                    successors.append((successor, prob))
                return DiscreteDistribution(successors)

    def simulate_run(self, policy: Dict[State, DiscreteDistribution[Action]]) -> Tuple[float, float, List[ActionName]]:
        s = self.initial_state
        actions = []
        while not self.terminal_state(s):
            a = policy[s].sample()
            s = self.transition_probabilities(s, a).sample()
            actions.append(a.name)
        return s.pain, sum([a.monetary_cost for a in s.actions_taken]), actions

    def transition_costs(self, s: State, a: Action) -> Tuple:
        if a.name == ActionName.Discharge:
            return float(s.pain), float(0)
        else:
            return float(0), float(a.monetary_cost)

    def terminal_state(self, s: State) -> bool:
        if [a for a in s.actions_taken if a.name == ActionName.Discharge]:
            return True
        else:
            return False


def terminal_state(s: State) -> bool:
    if [a for a in s.actions_taken if a.name == ActionName.Discharge]:
        return True
    else:
        return False


def transition_costs(s: State, a: Action) -> Tuple:
    if a.name == ActionName.Discharge:
        return float(s.pain), float(0)
    else:
        return float(0), float(a.monetary_cost)