from abc import ABC, abstractmethod
from enum import Enum
from typing import List
from skdecide import DiscreteDistribution

"""
Author: Charles Evans
Email: u6942700@anu.edu.au
This is my own work, and forms part of my artefact contribution for COMP3770, Semester 1, 2021.
"""


class State(ABC):
    """
    Abstract class for a C-SSP state.
    """
    id: int

    def __init__(self, identification):
        self.id = identification


class ActionName(Enum):
    """
    Abstract class for an ActionName enum.
    """
    pass


class Action(ABC):
    """
    Abstract class for a C-SSP action.
    """
    name: ActionName

    def __init__(self, name):
        self.name = name


class MorallyConsequentialCsspEnv(ABC):
    """
    Abstract class for a C-SSP environment.
    Defines necessary fields/methods to represent the tuple <S, s0, G, A, P, \overrightarrow{C}, \overrightarrow{\hat{c}}>.
    """

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
        """
        Returns the list of applicable actions (i.e. can legally be applied) from some state s.

        Args:
            s: a state
        Returns:
            List[Action]
        """
        pass

    @abstractmethod
    def transition_probabilities(self, s: State, a: Action) -> DiscreteDistribution[State]:
        """
        Returns a DiscreteDistribution over possible successor states, if taking action a from state s.

        Args:
            s: a state
            a: an action

        Returns:
            DiscreteDistribution[State]
        """
        pass

    @abstractmethod
    def transition_costs(self, s: State, a: Action) -> List[float]:
        """
        Returns the cost vector for taking action a in state s: i-th element of the return list corresponds to C_i(s,a)

        Args:
            s: a state
            a: an action

        Returns:
            List[float]
        """
        pass

    @abstractmethod
    def terminal_state(self, s: State) -> bool:
        """
        Returns true/false representing whether the state s is in G (the set of goal states).

        Args:
            s: a state

        Returns:
            bool

        """
        pass
