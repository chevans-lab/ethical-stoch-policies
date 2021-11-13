from env.mc_cssp_env import MorallyConsequentialCsspEnv, State, Action, ActionName

from typing import Set, List
from skdecide import DiscreteDistribution
from itertools import chain, product, combinations

"""
Author: Charles Evans
Email: u6942700@anu.edu.au
This is my own work, and forms part of my artefact contribution for COMP3770, Semester 1, 2021.
"""


class MedicActionName(ActionName):
    """
    Enum of action names for the Automated Medic C-SSP. Discharge refers to the action of discharging a patient,
    All other values are names of the painkiller that will be administered by the corresponding action.
    """
    A = 1
    B = 2
    C = 3
    Discharge = 4


class MedicAction(Action):
    """
    An action in the Automated Medic C-SSP. Constains:
    - name: a unique enum value (inherited from Action)
    - monetary_cost: For all painkiller-applying actions, the monetary cost of the painkiller. 0 for discharge action.
    - effect: A DiscreteDistribution over the amount of pain reduction the painkiller will provide,
      for painkiller-applying actions.
        - If the sampled reduction is greater than the pain level in the state where the action is being applied,
          the successor state will have pain 0 (will not go negative).
    """
    name: MedicActionName
    monetary_cost: int
    effect: DiscreteDistribution[int]

    def __init__(self, name, monetary_cost, effect):
        super().__init__(name)
        self.monetary_cost = monetary_cost
        self.effect = effect


class MedicState(State):
    """
    A state in the Automated Medic C-SSP. Contains:
    - id: a unique integer ID (inherited from State)
    - pain: the current self-reported pain of the patient being treated
    - actions_taken: a set of actions already applied (needed to track what medications have been administered already)
    """

    pain: int
    actions_taken: Set[MedicAction]

    def __init__(self, identification, pain, actions_taken):
        super().__init__(identification)
        self.pain = pain
        self.actions_taken = actions_taken


class MedicEnv(MorallyConsequentialCsspEnv):
    """
    The Automated Medic C-SSP environment, extends MorallyConsequentialCsspEnv. See report for more information.
    Basic setup is the following:

    - Medic chooses a particular painkiller to administer to a patient at each timestep, or else chooses to discharge
      them from its care.
    - Each medication can be given once only.
    - Applying a medication has only a very small primary cost, but incurs secondary monetary cost
    - Discharging incurs the patient's remaining pain as a primary cost, but no secondary cost
    - Objective is to minimise expected primary cost, so remaining pain upon discharge.
    - Expected monetary cost will also be upper bounded.
    """

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
                 num_secondary_costs):
        super().__init__(state_space,
                         initial_state,
                         goal_states,
                         action_space,
                         secondary_cost_bounds,
                         num_secondary_costs)

    def applicable_actions(self, s: MedicState) -> List[MedicAction]:
        """
        Returns the list of applicable actions (i.e. can legally be applied) from some state s.

        Args:
            s: a state
        Returns:
            List[Action]
        """
        if not [a for a in s.actions_taken if a.name == MedicActionName.Discharge]:  # if patient not discharged yet
            return [a for a in self.action_space if a not in s.actions_taken]  # all unadministered meds, plus discharge
        else:
            return []  # patient discharged, no actions applicable

    def transition_probabilities(self, s: MedicState, a: MedicAction) -> DiscreteDistribution[MedicState]:
        """
        Returns a DiscreteDistribution over possible successor states, if taking action a from state s.

        Args:
            s: a state
            a: an action

        Returns:
            DiscreteDistribution[State]
        """

        actions_taken = {a for a in s.actions_taken}
        actions_taken.add(a)
        if a in self.applicable_actions(s):
            if a.name == MedicActionName.Discharge:  # deterministic transition to goal state if action is a discharge
                successor = [state for state in self.state_space if state.pain == s.pain and state.actions_taken == actions_taken][0]
                return DiscreteDistribution([(successor, 1.0)])
            else:  # action is a painkiller administration
                pain_distribution = DiscreteDistribution([(max(s.pain - e, 0), prob) for e, prob in a.effect.get_values()])
                successors = []
                # Calculates a distribution over successors given the possible pain reduction outcomes of the painkiller
                for pain, prob in pain_distribution.get_values():
                    successor = [state for state in self.state_space if state.pain == pain and state.actions_taken == actions_taken][0]
                    successors.append((successor, prob))
                return DiscreteDistribution(successors)

    def transition_costs(self, s: MedicState, a: MedicAction) -> List[float]:
        """
        Returns the cost vector for taking action a in state s: i-th element of the return list corresponds to C_i(s,a)

        Args:
            s: a state
            a: an action

        Returns:
            List[float]
        """
        if a.name == MedicActionName.Discharge:
            # Incurs patient's remaining pain as primary cost, no secondary cost
            return [float(s.pain), float(0)]
        else:
            # Incurs small positive primary cost, monetary value of the painkiller as secondary cost
            return [float(0.001), float(a.monetary_cost)]

    def terminal_state(self, s: MedicState) -> bool:
        """
        Returns true/false representing whether the state s is in G (the set of goal states).

        Args:
            s: a state

        Returns:
            bool

        """
        if [a for a in s.actions_taken if a.name == MedicActionName.Discharge]:
            return True
        else:
            return False


def terminal_state(s: MedicState) -> bool:
    """
    Returns true/false representing whether the state s is in G (the set of goal states). Duplicates the instance
    method above as a standalone function as it is used in the construction of instances.

    Args:
        s: a state

    Returns:
        bool

    """
    if [a for a in s.actions_taken if a.name == MedicActionName.Discharge]:
        return True
    else:
        return False

def construct_instance(name: str) -> MedicEnv:
    """
    Constructs and returns a MedicEnv instance.

    Args:
        name: The name of the instance that should be constructed and returned.
    Returns:
        MedicEnv
    """
    if name == 'medic_small':

        # Three available painkillers
        action_list = [MedicAction(MedicActionName.B, 600, DiscreteDistribution([(3, 0.25), (5, 0.25), (6, 0.5)])),
                       MedicAction(MedicActionName.A, 1000, DiscreteDistribution([(5, 0.25), (6, 0.25), (10, 0.5)])),
                       MedicAction(MedicActionName.C, 500, DiscreteDistribution([(0, 0.2), (5, 0.8)])),
                       MedicAction(MedicActionName.Discharge, 0, None)]

        # Constructing all possible states
        action_combinations = [set(c) for c in
                               chain.from_iterable(combinations(action_list, r) for r in range(len(action_list) + 1))]
        state_space = [MedicState(i, *params) for i, params in enumerate(product(list(range(11)), action_combinations))]

        # Filtering state space to find the goal state list
        goal_states = [state for state in state_space if terminal_state(state)]

        # Identifying initial state (remaining pain=10 and no painkillers administered yet)
        initial_state = [state for state in state_space if state.pain == 10 and len(state.actions_taken) == 0][0]

        # Expected spending on painkillers in a policy execution must be less than $1200
        secondary_cost_bounds = [1200]

        return MedicEnv(state_space,
                        initial_state,
                        goal_states,
                        action_list,
                        secondary_cost_bounds,
                        len(secondary_cost_bounds))
