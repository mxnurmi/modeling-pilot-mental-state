from pomdp.domain import * # TODO: make explicit
from pomdp.models import * # same

import pomdp_py

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
        env = pomdp_py.Environment(init_true_state,  # these as well
                                   TransitionModel(n, k),
                                   RewardModel())
        super().__init__(agent, env, name="PlaneProblem")