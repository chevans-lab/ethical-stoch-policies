from typing import Set, List, Tuple, Dict
from ethical_cssp_env import EthicalCsspEnv, State, Action, ActionName

from skdecide import DiscreteDistribution


class DoctorActionName(ActionName):
    A = 1
    B = 2
    C = 3
    Discharge = 4


class DoctorAction(Action):
    name: DoctorActionName
    monetary_cost: int
    effect: DiscreteDistribution[int]

    def __init__(self, name, monetary_cost, effect):
        super().__init__(name)
        self.monetary_cost = monetary_cost
        self.effect = effect


class DoctorState(State):
    pain: int
    actions_taken: Set[DoctorAction]

    def __init__(self, identification, pain, actions_taken):
        super().__init__(identification)
        self.pain = pain
        self.actions_taken = actions_taken


class DoctorEnv(EthicalCsspEnv):

    state_space: List[DoctorState]
    initial_state: DoctorState
    goal_states: List[DoctorState]
    action_space: List[DoctorAction]

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

    def applicable_actions(self, s: DoctorState) -> List[DoctorAction]:
        if not [a for a in s.actions_taken if a.name == DoctorActionName.Discharge]:
            return [a for a in self.action_space if a not in s.actions_taken and a.name != "terminal"]
        else:
            return []

    def transition_probabilities(self, s: DoctorState, a: DoctorAction) -> DiscreteDistribution[DoctorState]:
        actions_taken = {a for a in s.actions_taken}
        actions_taken.add(a)
        if a in self.applicable_actions(s):
            if a.name == DoctorActionName.Discharge:
                successor = [state for state in self.state_space if state.pain == s.pain and state.actions_taken == actions_taken][0]
                return DiscreteDistribution([(successor, 1.0)])
            else:
                pain_distribution = DiscreteDistribution([(max(s.pain - e, 0), prob) for e, prob in a.effect.get_values()])
                successors = []
                for pain, prob in pain_distribution.get_values():
                    successor = [state for state in self.state_space if state.pain == pain and state.actions_taken == actions_taken][0]
                    successors.append((successor, prob))
                return DiscreteDistribution(successors)

    def simulate_run(self, policy: Dict[DoctorState, DiscreteDistribution[DoctorAction]]) -> Tuple[float, List[float], List[DoctorActionName]]:
        s: DoctorState = self.initial_state
        actions = []
        while not self.terminal_state(s):
            a = policy[s].sample()
            s = self.transition_probabilities(s, a).sample()
            actions.append(a.name)
        return float(s.pain), [sum([a.monetary_cost for a in s.actions_taken])], actions

    def transition_costs(self, s: DoctorState, a: DoctorAction) -> List[float]:
        if a.name == DoctorActionName.Discharge:
            return [float(s.pain), float(0)]
        else:
            return [float(0.001), float(a.monetary_cost)]

    def terminal_state(self, s: DoctorState) -> bool:
        if [a for a in s.actions_taken if a.name == DoctorActionName.Discharge]:
            return True
        else:
            return False


def terminal_state(s: DoctorState) -> bool:
    if [a for a in s.actions_taken if a.name == DoctorActionName.Discharge]:
        return True
    else:
        return False


def transition_costs(s: DoctorState, a: DoctorAction) -> Tuple:
    if a.name == DoctorActionName.Discharge:
        return float(s.pain), float(0)
    else:
        return float(0.001), float(a.monetary_cost)