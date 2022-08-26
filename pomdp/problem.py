import copy

import pomdp_py

from pomdp.domain import * # TODO: make explicit
from pomdp.models import * # same



class PlaneProblemEnvironment(pomdp_py.Environment):
    """"""
    def __init__(self, init_true_state, n, k):
        """
        Args:
            init_true_state (PlaneState): the starting plane state

        """

        reward_model = RewardModel()
        transition_model = TransitionModel(n, k)

        super().__init__(init_true_state,
                        transition_model,
                        reward_model)
        

    def state_transition(self, action, execute=True, override_state=None):
        """state_transition(self, action, execute=True, **kwargs)
        Overriding parent class function.
        Simulates a state transition given `action`. If `execute` is set to True,
        then the resulting state will be the new current state of the environment.
        Args:
            action (Action): action that triggers the state transition
            execute (bool): If True, the resulting state of the transition will
                            become the current state.
        Returns:
            float or tuple: reward as a result of `action` and state
            transition, if `execute` is True (next_state, reward) if `execute`
            is False.
        """
        #assert robot_id is not None, "state transition should happen for a specific robot"
        if override_state == None:
            next_state = copy.deepcopy(self.state)
            next_state = self.transition_model.sample(self.state, action)
            reward = self.reward_model.sample(self.state, action, next_state)
        if execute:
            self.apply_transition(next_state)
            return reward
        else:
            return next_state, reward        


# Hmm, this can be split to separate functions by calling pomdp_py.POMDP directly with different parts
# TODO: Update so that we create the problem environment here
class PlaneProblem(pomdp_py.POMDP):
    """
    PlaneProblem class is a wrapper
    """

    def __init__(self, n, k, init_true_state, init_belief):
        """init_belief is a Distribution."""

        agent = pomdp_py.Agent(init_belief,
                               PolicyModel(n, k),
                               TransitionModel(n, k),
                               ObservationModel(),
                               RewardModel())
        # TODO: you should produce a custom env here that allows us to override the transition sometimes to match the simulator
        env = PlaneProblemEnvironment(init_true_state, n, k)
        super().__init__(agent, env, name="PlaneProblem")