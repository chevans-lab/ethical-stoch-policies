from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Tuple
from skdecide import DiscreteDistribution


class State(ABC):
    id: int

    def __init__(self, identification):
        self.id = identification


class ActionName(Enum):
    pass


class Action(ABC):
    name: ActionName

    def __init__(self, name):
        self.name = name


class MorallyConsequentialCsspEnv(ABC):

    state_space: List[State]
    initial_state: State
    goal_states: List[State]
    action_space: List[Action]
    secondary_cost_bounds = List[float]
    num_secondary_costs = int

    def __init__(self,
                 state_space,
                 initial_state,
                 goal_states,
                 action_space,
                 secondary_cost_bounds,
                 num_secondary_costs):
        self.state_space = state_space
        self.initial_state = initial_state
        self.goal_states = goal_states
        self.action_space = action_space
        self.secondary_cost_bounds = secondary_cost_bounds
        self.num_secondary_costs = num_secondary_costs

    @abstractmethod
    def applicable_actions(self, s: State) -> List[Action]:
        pass

    @abstractmethod
    def transition_probabilities(self, s: State, a: Action) -> DiscreteDistribution[State]:
        pass

    @abstractmethod
    def simulate_run(self, policy: Dict[State, DiscreteDistribution[Action]]) -> Tuple[List[float], List[ActionName]]:
        pass

    @abstractmethod
    def transition_costs(self, s: State, a: Action) -> List[float]:
        pass

    @abstractmethod
    def terminal_state(self, s: State) -> bool:
        pass



