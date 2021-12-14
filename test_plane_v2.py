import pomdp_py
#from pomdp_problems.tiger import tiger_problem as tp
import random
from pomdp_py.utils import TreeDebugger
import numpy as np
import copy

EPSILON = 1e-3
START_FUEL = 5

# https://h2r.github.io/pomdp-py/html/_modules/pomdp_problems/tag/domain/action.html#TagAction
class PlaneAction(pomdp_py.Action):
    def __init__(self, name):
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
    def __init__(self, location, wind, fuel):
        self.location = location
        self.wind = wind
        self.fuel = fuel

    def __hash__(self):
        return hash((self.location, self.wind, self.fuel))

    def __eq__(self, other):
        if isinstance(other, PlaneState):
            # checks if the other state has identical name to this
            return self.location == other.location\
                and self.wind == other.wind\
                and self.fuel == other.fuel
        return False

    def __str__(self):
        return 'State(%s| %s, %s)' % (str(self.location),
                                       str(self.wind),
                                       str(self.fuel)
                                       )

    def __repr__(self):
        return str(self)


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
                return 0.7
            else:
                return 0.3
        else:
            if next_state != self.sample(state, action):
                return EPSILON
            else:
                return 1 - EPSILON

    def sample(self, state, action):
        if state.fuel < 1:
            return PlaneState("Linköping", True, START_FUEL) # reset

        if action.name == "land":
            if state.location == "Malmö" or state.wind == False:
                return PlaneState("Linköping", True, START_FUEL) # reset
            else:
                return PlaneState(state.location, state.wind, state.fuel - 1) # NOTE: if we try to land wind does not change
        if action.name == ("wait-wind"):
            windy_state = PlaneState("Linköping", True, state.fuel - 1)
            non_windy_state = PlaneState("Linköping", False, state.fuel - 1)
            return random.choices([windy_state, non_windy_state], weights=[0.7, 0.3], k=1)[0]
        if action.name == ("change-airport"):
            return PlaneState("Malmö", False, state.fuel - 3)

    def argmax(self, state, action, normalized=False, **kwargs):
        """Returns the most likely next state"""
        return self.sample(state, action)


# Policy Model
class PolicyModel(pomdp_py.RandomRollout):
    """This is an extremely dumb policy model; To keep consistent
    with the framework."""
    # A stay action can be added to test that POMDP solver is
    # able to differentiate information gathering actions.
    ACTIONS = {PlaneAction(s) for s in {"wait-wind", "change-airport", "land"}}

    def sample(self, state, **kwargs):
        return random.sample(self.get_all_actions(), 1)[0]

    def get_all_actions(self, **kwargs):
        return PolicyModel.ACTIONS


# Observation model
class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self, noise=0.15):
        self.noise = noise

    def probability(self, observation, next_state, action):
        if action.name == "wait-wind":
            if observation.wind == next_state.wind:  
                return 1.0 - self.noise # correct wind
            else:
                return self.noise # incorrect wind
        else:
            if observation.wind is None:
                return 1.0 - EPSILON  # expected to receive no observation
            else:
                return EPSILON

    def sample(self, next_state, action):
        if action.name == "wait-wind":
            thresh = 1.0 - self.noise

            if random.uniform(0,1) < thresh:
                return PlaneObservation(next_state.wind)
            else:
                return PlaneObservation(not next_state.wind)

        else:
            return PlaneObservation(None)


class RewardModel(pomdp_py.RewardModel):
    def _reward_func(self, state, action):
        if state.fuel < 1:
            return -100
        if action.name == "wait-wind": # Small punishment for waiting
            return -1
        elif action.name == "change-airport": # Should always have a punishment for changing airport
            return -25
        elif action.name == "land":
            if state.location == "Malmö" or state.wind == False:
                return 50
            else:
                return -10 # punish for trying to land when not able to

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

def generate_random_state():
    location = "Linköping"
    return PlaneState(location, True, START_FUEL)


def generate_init_belief(num_particles):
    particles = []
    for _ in range(num_particles):
        particles.append(generate_random_state())

    return pomdp_py.Particles(particles)


def test_planner(plane_problem, planner, nsteps=5, debug_tree=False):
    """
    Runs the action-feedback loop of Tiger problem POMDP

    Args:
        plane_problem (PlaneProblem): an instance of the plane problem.
        planner (Planner): a planner
        nsteps (int): Maximum number of steps to run this loop.
    """

    for i in range(nsteps):
        true_state = copy.deepcopy(plane_problem.env.state)
        action = planner.plan(plane_problem.agent)

        print("==== Step %d ====" % (i+1))
        print("True state: %s" % true_state)
        print("Belief: %s" % str(plane_problem.agent.cur_belief))
        print("Action: %s" % str(action))
        print("Reward: %s" % str(plane_problem.env.reward_model.sample(
            plane_problem.env.state, action, None)))
        print("\n")

        
        # TODO: What is this needed for?
        env_reward = plane_problem.env.state_transition(action, execute=True)
        
        real_observation = plane_problem.env.provide_observation(
                plane_problem.agent.observation_model, action)
        plane_problem.agent.update_history(action, real_observation)
        planner.update(plane_problem.agent, action, real_observation)
        
        
        action = planner.plan(plane_problem.agent)


        """
        if debug_tree:
            from pomdp_py.utils import TreeDebugger
            dd = TreeDebugger(plane_problem.agent.tree)
            import pdb
            pdb.set_trace()

        # Let's create some simulated real observation; Update the belief
        # Creating true observation for sanity checking solver behavior.
        # In general, this observation should be sampled from agent's observation model.
        real_observation = PlaneObservation(plane_problem.env.state.wind)
        print(">> Observation: %s" % real_observation)
        plane_problem.agent.update_history(action, real_observation)

        # If the planner is POMCP, planner.update also updates agent belief.

        print("tree is below")
        print(plane_problem.agent.tree[PlaneAction("wait-wind")])
        print("action:", action)
        print("real_observation:", real_observation)

        planner.update(plane_problem.agent, action, real_observation) # Problem here

        print("We've made it")

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
        """


def main():
    init_true_state = PlaneState("Linköping", True, START_FUEL)
    #init_belief = pomdp_py.Histogram({PlaneState("Linköping", False): 1.0 - EPSILON, PlaneState("Linköping", True): EPSILON})
    init_belief = generate_init_belief(200)

    plane_problem = PlaneProblem(init_true_state, init_belief)

    #print("** Testing value iteration **")
    #vi = pomdp_py.ValueIteration(horizon=3, discount_factor=0.95)
    #test_planner(plane_problem, vi, nsteps=3) # HERE

    # Reset agent belief
    #init_belief = pomdp_py.Histogram({PlaneState("Linköping", False): 1.0 - EPSILON, PlaneState("Linköping", True): EPSILON})
    #init_belief = generate_init_belief(200)
    #plane_problem.agent.set_belief(init_belief, prior=False)

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
    #plane_problem.agent.tree = None

    print("** Testing POMCP **")
    #plane_problem.agent.set_belief(pomdp_py.Particles.from_histogram(
        #init_belief, num_particles=100), prior=True)
    init_belief = generate_init_belief(num_particles=100)
    plane_problem.agent.set_belief(init_belief)
    init_state = PlaneState("Linköping", False, START_FUEL)
    #plane_problem = PlaneProblem(init_state, init_belief)
    pomcp = pomdp_py.POMCP(max_depth=5, discount_factor=0.95,
                           num_sims=1000, exploration_const=50,
                           rollout_policy=plane_problem.agent.policy_model,
                           show_progress=True, pbar_update_interval=500)

    test_planner(plane_problem, pomcp, nsteps=5) 
    # TODO: check the tree and how to interpret it 
    TreeDebugger(plane_problem.agent.tree).pp


if __name__ == '__main__':
    main()
