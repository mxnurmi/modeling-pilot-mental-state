# %%

import pomdp_py
import random
import copy

from pomdp_py.utils import TreeDebugger

from visual import visualize_plane_problem
from stress_functions import stress_function

EPSILON = 1e-3
START_FUEL = 35 # also max fuel
STRESS = 0
MALMO_LOCATION = (7, 9)  # (7, 9)
LINKOPING_LOCATION = (21, 8)  
SIZE = (31, 15)
WIND_PROB = 1

MALMO_LOCATION = (1,1)  # (7, 9)
LINKOPING_LOCATION = (5,5)  
SIZE = (13,13)

# TODO: What generates stress?
# -> every turn wind doesn't change -> more stress
# -> every turn fuel gets lower -> more stress
# -> uncertainty -> more stress

# TODO: What constitutes as cognitive workload?

# Are stress and cognitive workload only external or internal states of the agent?

# We should have a global variable STRESS which is a function of (abs(p(event) x r(event)), kun r < 0)
# This then affects Noise level
# For this we'll need p for each event with negative reward


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


class MoveAction(PlaneAction):
    EAST = (1, 0)  # x is horizontal; x+ is right. y is vertical; y+ is up.
    WEST = (-1, 0)
    NORTH = (0, 1)  # are these correct? -> changed from original
    SOUTH = (0, -1)  # are these correct? -> changed from original

    def __init__(self, motion, name):
        if motion not in {MoveAction.EAST, MoveAction.WEST,
                          MoveAction.NORTH, MoveAction.SOUTH}:
            raise ValueError("Invalid move motion %s" % motion)
        self.motion = motion
        super().__init__("move-%s" % str(name))


MoveEast = MoveAction(MoveAction.EAST, "EAST")
MoveWest = MoveAction(MoveAction.WEST, "WEST")
MoveNorth = MoveAction(MoveAction.NORTH, "NORTH")
MoveSouth = MoveAction(MoveAction.SOUTH, "SOUTH")

# waiting also checks for wind


class WaitAction(PlaneAction):
    def __init__(self):
        super().__init__("check-wind")


class LandAction(PlaneAction):
    def __init__(self):
        super().__init__("land")

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

    def __init__(self, n, k):
        self._n = n
        self._k = k

    def probability(self, next_state, state, action):
        # TODO: we should have transition between differing wind states
        # In the tiger example there are always two states and opening a door will corrspond
        # to (state, action, next_state) which is 0.5 always as long as action is opening
        # In this transitionmodel we can just explicitly give out all of the probabilities instead

        #if isinstance(action, WaitAction) or isinstance(action, MoveAction):
        #    if next_state.wind == False:
        #        return WIND_PROB
        #    else:
        #        return 1 - WIND_PROB
        #else:
        if next_state != self.sample(state, action):
            return EPSILON
        else:
            return 1 - EPSILON

    def sample(self, state, action):
        # return PlaneState("LINKÖPING", state.wind, state.fuel)
        # TODO: make a better structure for this
        if state.location == "landed" or state.location == "crashed":
            # nothing changes
            return PlaneState(state.location, state.wind, state.fuel)

        if state.fuel < 1:
            return PlaneState("crashed", True, state.fuel)

        if isinstance(action, LandAction):
            # TODO: FIX based on new locations
            if (state.location == MALMO_LOCATION) or (state.location == LINKOPING_LOCATION and state.wind == False):
                return PlaneState("landed", True, state.fuel)
            else:
                # NOTE: if we try to incorrectly land wind does not change ?
                return PlaneState(state.location, state.wind, state.fuel - 1)

        wind_state = random.choices([True, False], weights=[
                                    WIND_PROB, 1-WIND_PROB], k=1)[0]

        if isinstance(action, WaitAction):
            #windy_state = PlaneState(state.location, True, state.fuel - 1)
            return PlaneState(state.location, wind_state, state.fuel)
            # return random.choices([windy_state, non_windy_state], weights=[0.7, 0.3], k=1)[0]

        if isinstance(action, MoveAction):
            # TODO: We should handle actions that go beyond the task domain with min + max
            #new_location = (state.location[0] + action.motion[0], state.location[1] + action.motion[1])

            new_location = (max(0, min(state.location[0] + action.motion[0], self._n - 1)),
                            max(0, min(state.location[1] + action.motion[1], self._k - 1)))

            return PlaneState(new_location, wind_state, state.fuel - 1)

        # backup. TODO: Better solution
        return PlaneState(state.location, state.wind, state.fuel)

    def argmax(self, state, action, normalized=False, **kwargs):
        """Returns the most likely next state"""
        return self.sample(state, action)


class PolicyModel(pomdp_py.RolloutPolicy):
    def __init__(self, n, k):
        # old: {"wait-wind", "change-airport", "land"}}
        self._move_actions = {MoveEast, MoveWest, MoveNorth, MoveSouth}
        self._other_actions = {WaitAction(), LandAction()}
        self._all_actions = self._move_actions | self._other_actions
        self._n = n
        self._k = k

    def sample(self, state, **kwargs):
        return random.sample(self.get_all_actions(), 1)[0]

    def get_all_actions(self, **kwargs):
        state = kwargs.get("state", None)
        if state is None:
            return self._all_actions
        else:
            motions = set(self._move_actions)

            if state.location != "crashed" and state.location != "landed":
                plane_x, plane_y = state.location
                if plane_x == self._k - 1:
                    motions.remove(MoveWest)
                if plane_y == 0:
                    motions.remove(MoveNorth)
                if plane_y == self._n - 1:
                    motions.remove(MoveSouth)
                if plane_x == 0:
                    motions.remove(MoveEast)  # TODO: correct?

            #print(motions | self._other_actions)
            return motions | self._other_actions

    def probability(self, action, state, normalized=False, **kwargs):
        raise NotImplementedError

    def argmax(self, state, normalized=False, **kwargs):
        """Returns the most likely reward"""
        raise NotImplementedError


class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self, noise=0):
        self.noise = noise

    def probability(self, observation, next_state, action):
        if isinstance(action, WaitAction) and (next_state.location != "landed") and (next_state.location != "crashed"):
            if observation.wind == next_state.wind:
                return 1.0 - self.noise  # correct wind
            else:
                return self.noise  # incorrect wind
        else:
            if observation.wind is None:
                return 1.0 - EPSILON  # expected to receive no observation
            else:
                return EPSILON

    def sample(self, next_state, action):
        if isinstance(action, WaitAction) and (next_state.location != "landed") and (next_state.location != "crashed"):
            thresh = 1.0 - self.noise

            if random.uniform(0, 1) < thresh:
                return PlaneObservation(next_state.wind)
            else:
                return PlaneObservation(not next_state.wind)
        else:
            return PlaneObservation(None)


class RewardModel(pomdp_py.RewardModel):
    def __init__(self):
        self._max_punish = -10000

    # TODO: make sure we reward if landed is next state
    def _reward_func(self, state, action):
        if state.location == "crashed":
            return self._max_punish
        elif isinstance(action, WaitAction):  # Small punishment for waiting
            return -2
        elif isinstance(action, MoveAction):
            return 0  # no punishment for moving
        elif isinstance(action, LandAction):
            if (state.location == MALMO_LOCATION): #or (state.location == LINKOPING_LOCATION and state.wind == False):
                return 1000
            else:
                return -5  # punish for trying to land when not able to
        return 0

    def sample(self, state, action, next_state):
        # deterministic
        return self._reward_func(state, action)

    def max_punishment(self):
        return self._max_punish

    # how to return max reward for normalization?


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


def generate_random_state():
    location = LINKOPING_LOCATION
    return PlaneState(location, True, START_FUEL)


def generate_init_belief(num_particles):
    particles = []
    for _ in range(num_particles):
        particles.append(generate_random_state())

    return pomdp_py.Particles(particles)


# TODO
# Make wind something that comes from a distribtuion and set a landing time variable
# if wind goes over threshold landing must be aborted

# TODO: Need a way to define:
# - Bunch of models
# - Bunch of training scenarios that can be each input to each model
# - Bunch of actual scenarios that can be each input to each model

# 1) tuulen nopeus tulee jostakin jatkuva-arvoisesta funktiosta, kuten sinikäyrä tmv.
# 2) jokaisessa trialissa arvotaan skenaarion parametrit, jolloin tuulen nopeus samplataan jostakin jakaumasta, ja tuon jakauman observoitu hajonta on stressi;
# 3) odotettu negatiivinen reward on stressitaso.. tässä vain ideoita

# knots = 38 (dry or wet) knots = 25 (snow) knots = 20 (3mm water) knots = 15 (ice)
# crosswind = sivutuuli
# laskeutumista yritetään jostain korkeudesta?
# windspeed from here: https://weatherspark.com/h/d/80053/2021/1/22/Historical-Weather-on-Friday-January-22-2021-in-Linköping-Sweden#Figures-WindSpeed




def test_planner(plane_problem, planner, nsteps=5, debug_tree=False, size=None):
    """
    Runs the action-feedback loop of Plane problem POMDP

    Args:
        plane_problem (PlaneProblem): an instance of the plane problem.
        planner (Planner): a planner
        nsteps (int): Maximum number of steps to run this loop.
    """

    total_reward = 0
    stress_states_sine = []
    stress_states_normal = []

    # TODO: Change to either "landed" or "crashed" to be the terminal state
    true_location = LINKOPING_LOCATION
    i = 0

    while (true_location != "landed") and (true_location != "crashed"):
        true_state = copy.deepcopy(plane_problem.env.state)
        true_location = true_state.location

        action = planner.plan(plane_problem.agent)
        env_reward = plane_problem.env.state_transition(action, execute=True) #TODO: use this for sampling
        total_reward += env_reward

        print("==== Step %d ====" % (i+1))
        print("True state: %s" % true_state)
        print("Belief: %s" % str(plane_problem.agent.cur_belief))
        print("Action: %s" % str(action))
        print("Reward: %s" % str(env_reward))
        print("Reward (Cumulative): %s" % str(total_reward))
        if isinstance(planner, pomdp_py.POUCT):
            print("__num_sims__: %d" % planner.last_num_sims)
            print("__plan_time__: %.5f" % planner.last_planning_time)
        if isinstance(planner, pomdp_py.PORollout):
            print("__best_reward__: %d" % planner.last_best_reward)
        print("\n")


        n, k = size # TODO: size gets none by default, fix or do error handling
        txt_action = str(action)
        txt_reward = str(total_reward)

        text = """
                Action: %s \n
                Reward (cumulative): %s \n
               """ % (txt_action, txt_reward)

        # TODO: visualized action is now one step behind?
        visualize_plane_problem(width=n,
                        height=k,
                        plane_location=true_state.location,
                        airport_location1=MALMO_LOCATION,
                        airport_location2=LINKOPING_LOCATION,
                        plot=True
                        )

        if debug_tree == True:
            dd = TreeDebugger(plane_problem.agent.tree)
            print(dd.pp)
            print(dd.mbp)  # TODO: what is the "none" leaf?
            print("\n")

        real_observation = plane_problem.env.provide_observation(
            plane_problem.agent.observation_model, action)
        plane_problem.agent.update_history(action, real_observation)
        planner.update(plane_problem.agent, action, real_observation)


        # TODO: Make sure we always pick strongest belief (now pick first?)
        for belief in plane_problem.agent.cur_belief:
            belief_state = belief
            break

        stress_sine = stress_function("sine_wind_based", belief_state, start_fuel=START_FUEL)
        stress_normal = stress_function("normal_wind_based", belief_state, start_fuel=START_FUEL)

        stress_states_sine.append(stress_sine)
        stress_states_normal.append(stress_normal)

        i += 1
    
    print("\nSine stress")
    print(stress_states_sine)

    print("\nNormal stress")
    print(stress_states_normal)

    print("\n")
    print("=== DONE ===")


# TODO: Split the main function so that it can run any of the given functions given an input and model
def main():
    init_true_state = PlaneState(LINKOPING_LOCATION, True, START_FUEL)
    init_belief = generate_init_belief(50)

    n = SIZE[0]
    k = SIZE[1]

    plane_problem = PlaneProblem(n, k, init_true_state, init_belief)

    # prior = True seems to reset the belief state completely
    # (https://github.com/h2r/pomdp-py/blob/master/pomdp_py/framework/basics.pyx)
    # plane_problem.agent.set_belief(init_belief, prior=True)

    pomcp = pomdp_py.POMCP(max_depth=10, discount_factor=0.999,  # what does the discount_factor do?
                           num_sims=5000, exploration_const=3000,
                           rollout_policy=plane_problem.agent.policy_model,
                           show_progress=False, pbar_update_interval=100)
    test_planner(plane_problem, planner=pomcp, nsteps=10, size=(n, k))

# -- Why is the fuel situation uncertain if we incorrectly land? --
# -> Related to the fact that the model does not know when the problem truly resets


if __name__ == '__main__':
    main()

# TODO: The final stress model should be one where we sample from all the actions
# and the higher there is a chance for max negative reward, the more stress the agent
# has

# Stress should come from either high uncertainty with likely negative reward or
# high chance of negative reward

# TODO: Could we fix this by adding knowldge about malmö location??

# ==== OLD =====

# TODO: This is currently "wrong" and evaluates stress assuming current state. However we would
# want to evaluate stress over the expected reward over different actions.
# This would need evaluation over all outcomes as well? Maybe easiest way to do this would be by MCMC
# sampling (generate samples from posterior)


def compute_stress_old(plane_problem, action):
    # get all states agent is considering
    prob_dict = plane_problem.agent.cur_belief.get_histogram()
    stress_sum = 0

    for key in prob_dict:
        state = [key][0]
        probability = prob_dict[key]
        corresponding_reward = RewardModel().sample(state, action, next_state=None)
        if corresponding_reward < 0:
            stress = probability * corresponding_reward
        else:
            stress = 0
        stress_sum += stress

    return stress_sum