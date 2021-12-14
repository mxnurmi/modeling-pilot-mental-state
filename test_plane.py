import pomdp_py
#from pomdp_problems.tiger import tiger_problem as tp
import random
from pomdp_py.utils import TreeDebugger

EPSILON = 1e-3

# https://h2r.github.io/pomdp-py/html/_modules/pomdp_problems/tag/domain/action.html#TagAction
class PlaneAction(pomdp_py.Action):
    def __init__(self, name): # should this be something else than name??
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, PlaneAction):
            return self.name == other.name
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return "PlaneAction(%s)" % self.name

# https://h2r.github.io/pomdp-py/html/_modules/pomdp_problems/tag/domain/state.html#TagState
class PlaneState(pomdp_py.State):
    def __init__(self, location, fuel, wind):
        self.location = location
        self.fuel = fuel  
        self.wind = wind

    def __hash__(self):
        return hash((self.location, self.fuel, self.wind))

    def __eq__(self, other):
        if isinstance(other, PlaneState):
            # checks if the other state has identical name to this
            return self.location == other.location\
                and self.fuel == other.fuel\
                and self.wind == other.wind
        return False

    def __str__(self):
        return 'State(%s, %s | %s)' % (str(self.location),
                                       str(self.fuel),
                                       str(self.wind))

    def __repr__(self):
        return str(self)

    """
    def other(self):
        if self.name.endswith("left"):
            return TigerState("tiger-right")
        else:
            return TigerState("tiger-left")
    """


class PlaneObservation(pomdp_py.Observation):
    def __init__(self, wind):
        self.wind = wind

    def __hash__(self):
        return hash(self.wind)

    def __eq__(self, other):
        if isinstance(other, PlaneObservation):
            return self.wind == other.wind
        return False

    def __str__(self):
        return str(self.wind)

    def __repr__(self):
        return str(self)

class TransitionModel(pomdp_py.TransitionModel):
    def probability(self, next_state, state, action):
        # TODO: we should have transition between differing wind states
        # In the tiger example there are always two states and opening a door will corrspond
        # to (state, action, next_state) which is 0.5 always as long as action is opening
        # In this transitionmodel we can just explicitly give out all of the probabilities instead
        
        if action.name == ("wait-wind"):
            if next_state.wind == False:
                return 0.8
            else:
                return 0.2
        else:
            if next_state != self.sample(state, action):
                return EPSILON
            else:
                return 1 - EPSILON

    def sample(self, state, action):
        if state.location == "Malmö" or (state.location == "Linköping" and state.wind == False) or state.fuel == 0:
            return PlaneState("Linköping", 10, True) # reset
        if action.name == ("wait-wind"):
            windy_state = PlaneState("Linköping", state.fuel - 1, True) # location, fuel, wind
            non_windy_state = PlaneState("Linköping", state.fuel - 1, False) # location, fuel, wind
            return random.choices([windy_state, non_windy_state], weights=[0.8, 0.2], k=1)[0]
        if action.name == ("change-airport"):
            return PlaneState("Malmö", state.fuel - 5, False) 
            # if we open, transition to new random state
            # return random.choice(self.get_all_states())

    def argmax(self, state, action, normalized=False, **kwargs):
        """Returns the most likely next state"""
        return self.sample(state, action)


# Policy Model
class PolicyModel(pomdp_py.RandomRollout):
    """This is an extremely dumb policy model; To keep consistent
    with the framework."""
    # A stay action can be added to test that POMDP solver is
    # able to differentiate information gathering actions.
    ACTIONS = {PlaneAction(s) for s in {"wait-wind", "change-airport"}}

    def sample(self, state, **kwargs):
        return random.sample(self.get_all_actions(), 1)[0]

    def get_all_actions(self, **kwargs):
        return PolicyModel.ACTIONS


# Observation model
class ObservationModel(pomdp_py.ObservationModel):
    #def __init__(self, noise=0.15):
        #self.noise = noise

    def probability(self, observation, next_state, action):
        if action.name == "wait-wind":
            if observation.wind == next_state.wind:  # get the wind in next state
                return 1.0 - EPSILON
            else:
                return EPSILON

    def sample(self, next_state, action):
        if action.name == "wait-wind":
            return PlaneObservation(next_state.wind)


class RewardModel(pomdp_py.RewardModel):
    def _reward_func(self, state, action):
        if state.fuel == 0:
            return -100 # out of fuel
        elif action.name == "wait-wind": # Small punishment for waiting, but high risk if we run out of fuel
            return -1
        elif action.name == "change-airport": # Should always have a punishment for changing airport
            return -20

    def sample(self, state, action, next_state):
        # deterministic
        return self._reward_func(state, action)


# Hmm, this can be broken up by calling pomdp_py.POMDP directly with different parts
class PlaneProblem(pomdp_py.POMDP):
    """
    PlaneProblem class is a wrapper
    """

    def __init__(self, init_true_state, init_belief):
        """init_belief is a Distribution."""
        agent = pomdp_py.Agent(init_belief,
                               PolicyModel(),
                               TransitionModel(),
                               ObservationModel(),
                               RewardModel())
        env = pomdp_py.Environment(init_true_state,
                                   TransitionModel(),
                                   RewardModel())
        super().__init__(agent, env, name="PlaneProblem")

    @staticmethod
    def create():
        """
        Args:
            state (str): could be 'tiger-left' or 'tiger-right'; True state of the environment
            belief (float): Initial belief that the target is on the left; Between 0-1.
            obs_noise (float): Noise for the observation model (default 0.15)
        """
        init_true_state = PlaneState("Linköping", 10, False)
        init_belief = pomdp_py.Histogram({PlaneState("Linköping", 10, False): 1.0})
        plane_problem = PlaneProblem(init_true_state, init_belief)
        plane_problem.agent.set_belief(init_belief, prior=True)
        return plane_problem


def test_planner(plane_problem, planner, nsteps=3, debug_tree=False):
    """
    Runs the action-feedback loop of Tiger problem POMDP

    Args:
        plane_problem (PlaneProblem): an instance of the plane problem.
        planner (Planner): a planner
        nsteps (int): Maximum number of steps to run this loop.
    """
    for i in range(nsteps):
        action = planner.plan(plane_problem.agent)
        if debug_tree:
            from pomdp_py.utils import TreeDebugger
            dd = TreeDebugger(plane_problem.agent.tree)
            import pdb
            pdb.set_trace()

        print("==== Step %d ====" % (i+1))
        print("True state: %s" % plane_problem.env.state)
        print("Belief: %s" % str(plane_problem.agent.cur_belief))
        print("Action: %s" % str(action))
        print("Reward: %s" % str(plane_problem.env.reward_model.sample(
            plane_problem.env.state, action, None)))

        # Let's create some simulated real observation; Update the belief
        # Creating true observation for sanity checking solver behavior.
        # In general, this observation should be sampled from agent's observation model.
        real_observation = PlaneObservation(plane_problem.env.state.wind)
        print(">> Observation: %s" % real_observation)
        plane_problem.agent.update_history(action, real_observation)

        # If the planner is POMCP, planner.update also updates agent belief.
        planner.update(plane_problem.agent, action, real_observation)
        if isinstance(planner, pomdp_py.POUCT):
            print("Num sims: %d" % planner.last_num_sims)
            print("Plan time: %.5f" % planner.last_planning_time)

        if isinstance(plane_problem.agent.cur_belief, pomdp_py.Histogram):
            new_belief = pomdp_py.update_histogram_belief(plane_problem.agent.cur_belief,
                                                          action, real_observation,
                                                          plane_problem.agent.observation_model,
                                                          plane_problem.agent.transition_model)
            plane_problem.agent.set_belief(new_belief)

        if action.name.startswith("change-airport"):
            # Make it clearer to see what actions are taken until every time airport is changed
            print("\n")


def main():
    init_true_state = PlaneState("Linköping", 10, False)
    init_belief = pomdp_py.Histogram({PlaneState("Linköping", 10, False): 1.0})

    plane_problem = PlaneProblem(init_true_state, init_belief)

    #print("** Testing value iteration **")
    #vi = pomdp_py.ValueIteration(horizon=3, discount_factor=0.95)
    #test_planner(plane_problem, vi, nsteps=3) # HERE

    # Reset agent belief
    init_belief = pomdp_py.Histogram({PlaneState("Linköping", 10, False): 1.0})
    plane_problem.agent.set_belief(init_belief, prior=True)

    """ not needed?
    print("\n** Testing POUCT **")
    pouct = pomdp_py.POUCT(max_depth=3, discount_factor=0.95,
                           num_sims=4096, exploration_const=50,
                           rollout_policy=tiger_problem.agent.policy_model,
                           show_progress=True)
    test_planner(tiger_problem, pouct, nsteps=10)
    TreeDebugger(tiger_problem.agent.tree).pp
    """

    # Reset agent belief
    plane_problem.agent.set_belief(init_belief, prior=True)
    plane_problem.agent.tree = None

    print("** Testing POMCP **")
    plane_problem.agent.set_belief(pomdp_py.Particles.from_histogram(
        init_belief, num_particles=100), prior=True)
    pomcp = pomdp_py.POMCP(max_depth=3, discount_factor=0.95,
                           num_sims=1000, exploration_const=50,
                           rollout_policy=plane_problem.agent.policy_model,
                           show_progress=True, pbar_update_interval=500)
    test_planner(plane_problem, pomcp, nsteps=10)
    TreeDebugger(plane_problem.agent.tree).pp


if __name__ == '__main__':
    main()
