import pomdp_py
#from pomdp_problems.tiger import tiger_problem as tp
import random
import math
import numpy as np
import copy

from scipy.stats import norm
from pomdp_py.utils import TreeDebugger

EPSILON = 1e-3
START_FUEL = 5 # also max fuel
STRESS = 0

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
            # TODO: Change reset into non-changing state
            return PlaneState("Linköping", True, START_FUEL)  # reset 

        if action.name == "land":
            if state.location == "Malmö" or state.wind == False:
                # TODO: Change reset into non-changing state
                return PlaneState("Linköping", True, START_FUEL)  # reset
            else:
                # NOTE: if we try to land wind does not change
                return PlaneState(state.location, state.wind, state.fuel - 1)
        if action.name == ("wait-wind"):
            # print("probs:")
            #print(self.probability(next_state=PlaneState("Linköping", False, 5), state=state, action=action))
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
                return 1.0 - self.noise  # correct wind
            else:
                return self.noise  # incorrect wind
        else:
            if observation.wind is None:
                return 1.0 - EPSILON  # expected to receive no observation
            else:
                return EPSILON

    def sample(self, next_state, action):
        if action.name == "wait-wind":
            thresh = 1.0 - self.noise

            if random.uniform(0, 1) < thresh:
                return PlaneObservation(next_state.wind)
            else:
                return PlaneObservation(not next_state.wind)
        else:
            return PlaneObservation(None)


class RewardModel(pomdp_py.RewardModel):
    def __init__(self):
        self._max_punish = -100

    def _reward_func(self, state, action):

        if state.fuel < 1:
            return self._max_punish
        if action.name == "wait-wind":  # Small punishment for waiting
            return -1
        elif action.name == "change-airport":  # Should always have a punishment for changing airport
            return -25
        elif action.name == "land":
            if state.location == "Malmö" or state.wind == False:
                return 50
            else:
                return -10  # punish for trying to land when not able to

    def sample(self, state, action, next_state):
        # deterministic
        return self._reward_func(state, action)

    def max_punishment(self):
        return self._max_punish

    # how to return max reward for normalization?


# Hmm, this can be split to separate functions by calling pomdp_py.POMDP directly with different parts
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

def normalize_value(x, min_value, max_value):
    return ((x - min_value) / (max_value - min_value))


def stress_function(function_name, plane_state):

    #wind = plane_state.wind
    # to normalize:
    # https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
    fuel = plane_state.fuel

    if function_name == "sine_wind_based":
        x = random.randint(25, 45) # wind speed
        wind_estimation = math.sin(x)
        stress = (wind_estimation + (fuel/START_FUEL)) / 2
        return stress

    if function_name == "normal_wind_based":
        x = norm.rvs(loc=35, scale=5, size=1) 
        maxx = 55
        minx = 15
        x = max(min(x, maxx), minx) #limit min/max to 15/55 so we are unlikely to get values outside those
        x = normalize_value(x, minx, maxx) # normalize between min and max stress
        wind_estimation = x
        stress = (wind_estimation + (fuel/START_FUEL)) / 2
        return stress

    if function_name == "expected_negative_reward":
        # generate expected negative reward by sampling outcomes based on possible actions
        return 0



def test_planner(plane_problem, planner, nsteps=5, debug_tree=False):
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

    for i in range(nsteps):
        true_state = copy.deepcopy(plane_problem.env.state)
        action = planner.plan(plane_problem.agent)
        env_reward = plane_problem.env.state_transition(action, execute=True)
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

        if debug_tree == True:      
            dd = TreeDebugger(plane_problem.agent.tree)
            print(dd.pp)
            print(dd.mbp) # TODO: Why is there a "none" leaf?
            print("\n")

        real_observation = plane_problem.env.provide_observation(
            plane_problem.agent.observation_model, action)
        plane_problem.agent.update_history(action, real_observation)
        planner.update(plane_problem.agent, action, real_observation)

        #action = planner.plan(plane_problem.agent)
        # TODO: compute stress here?
        # We can just break the loop when we reach goal state??

        # TODO: Make sure we always pick stronges belief
        for belief in plane_problem.agent.cur_belief:
            belief_state = belief
            break

        stress_sine = stress_function("sine_wind_based", belief_state)
        stress_normal = stress_function("normal_wind_based", belief_state)

        stress_states_sine.append(stress_sine)
        stress_states_normal.append(stress_normal)

        if (((true_state.wind == False) or (true_state.location == 'Malmö')) and str(action) == 'land'):
            print("sine stress")
            print(stress_states_sine)

            print("normal stress")
            print(stress_states_normal)

            break


# TODO: Split the main function so that it can run any of the given functions given an input and model


def main():
    init_true_state = PlaneState("Linköping", True, START_FUEL)
    init_belief = generate_init_belief(200)
    plane_problem = PlaneProblem(init_true_state, init_belief)
    #init_belief = generate_init_belief(num_particles=100)

    # prior = True seems to reset the belief state completely
    # (https://github.com/h2r/pomdp-py/blob/master/pomdp_py/framework/basics.pyx)
    # plane_problem.agent.set_belief(init_belief, prior=True)

    pomcp = pomdp_py.POMCP(max_depth=2, discount_factor=0.95,
                           num_sims=500, exploration_const=50,
                           rollout_policy=plane_problem.agent.policy_model,
                           show_progress=True, pbar_update_interval=500)

    test_planner(plane_problem, planner=pomcp, nsteps=5)

# -- Why is the fuel situation uncertain if we incorrectly land? --
# -> Related to the fact that the model does not know when the problem truly resets

if __name__ == '__main__':
    main()

# TODO: The final stress model should be one where we sample from all the actions
# and the higher there is a chance for max negative reward, the more stress the agent
# has

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
