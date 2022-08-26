
import pomdp_py
import random

from math import dist

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)

from domain import PlaneState, PlaneObservation, LandAction, WaitAction, MoveAction, MoveEast, MoveNorth, MoveSouth, MoveWest

# Hacky way to get imports from parent folder
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import config
#print(config)

# TODO: Particle reinvigoration function. 
# Challenge: Authors (Silver, Veness 2010) count it as sim_amount/16.
# We could achieve this with gloabl variable?
class TransitionModel(pomdp_py.TransitionModel):

    def __init__(self, n, k):
        self._n = n
        self._k = k

    def probability(self, next_state, state, action):

        if next_state != self.sample(state, action):
            return config.EPSILON
        else:
            return 1 - config.EPSILON

    def sample(self, state, action):
        if state.position == "pre-flight":
            return PlaneState(config.LINKOPING_LOCATION, "takeoff", True, config.START_FUEL) 

        if state.position == "landed" or state.position == "crashed":
            # TODO: Which one of these is correct from the stress mdoeling perspective? Does resetting affect the complexity incorrectly etc.?
            #return PlaneState(state.coordinates, state.position, state.wind, state.fuel) # final state
            return PlaneState(config.LINKOPING_LOCATION, "pre-flight", True, config.START_FUEL)  # reset state

        if state.fuel < 1:
            return PlaneState(state.coordinates, "crashed", True, state.fuel)

        ## Add n/16 random particles (n = num_of_sim)

        #### ----\/ currently not in use \/ ------
        particles = config.NUM_SIMS/16
        particle_prob = particles/config.NUM_SIMS
        particle_prob = 0 # NOTE: override, TODO: make global variable
        add_particle = random.choices([True, False], weights=[
                            particle_prob, 1-particle_prob], k=1)[0]

        if add_particle == True:
            plane_x = random.randint(0, self._n - 1)
            plane_y = random.randint(0, self._k - 1)
            random_coordinates = (plane_x, plane_y)
            return PlaneState(random_coordinates, state.position, state.wind, state.fuel)
        #### ----  ^^^ currently not in use ^^^ ------

        # TODO: Wind should be estimated from previous wind?
        wind_state = random.choices([True, False], weights=[
                                    config.WIND_PROB, 1-config.WIND_PROB], k=1)[0]

        fuel_state = random.choices([state.fuel - 1, state.fuel - config.DUMB_AMOUNT], weights=[
                                    config.FUEL_PROB, 1-config.FUEL_PROB], k=1)[0]

        if isinstance(action, LandAction):
            # TODO: We should make it so that wind always prevents landing but only turns on near one of the airports?
            if ((state.coordinates == config.MALMEN_LOCATION and state.wind == True) or (state.coordinates == config.LINKOPING_LOCATION and state.wind == False)):
                if state.position=="landing":
                    successful_landing = random.choices([True, False], weights=[0.85, 0.15], k=1)[0] # TODO: add to config

                    if successful_landing == True:
                        return PlaneState(state.coordinates, "landed", False, state.fuel)
                    else:
                        crash_during_landing = random.choices([True, False], weights=[0.01, 0.99], k=1)[0] # TODO: add to config
                        if crash_during_landing == False:
                            return PlaneState(state.coordinates, "landing", wind_state, fuel_state)
                        else:
                            return PlaneState(state.coordinates, "crashed", True, state.fuel)

                elif state.position=="flying":
                    return PlaneState(state.coordinates, "landing", wind_state, state.fuel - 1)
            else:
                return PlaneState(state.coordinates, state.position, wind_state, state.fuel - 2)

        #if isinstance(action, WaitAction):
            #return PlaneState(state.coordinates, state.position, wind_state, fuel_state)

        successfull_takeoff = random.choices([True, False], weights=[0.9, 0.1], k=1)[0] # TODO: add to config

        if isinstance(action, MoveAction):
            # We handle actions that go beyond the task domain with min + max

            new_coordinates = (max(0, min(state.coordinates[0] + action.motion[0], self._n - 1)),
                            max(0, min(state.coordinates[1] + action.motion[1], self._k - 1)))

            if state.position == "takeoff":
                if successfull_takeoff:
                    return PlaneState(new_coordinates, "flying", wind_state, fuel_state)
                else:
                    crash_during_takeoff = random.choices([True, False], weights=[0.01, 0.99], k=1)[0] # TODO: add to config
                    if crash_during_takeoff == False:
                        return PlaneState(state.coordinates, "takeoff", wind_state, fuel_state)
                    else:
                        return PlaneState(state.coordinates, "crashed", True, state.fuel)

            return PlaneState(new_coordinates, state.position, wind_state, fuel_state)

        # backup. TODO: Better solution?
        return PlaneState(state.coordinates, state.position, state.wind, fuel_state)

    def argmax(self, state, action, normalized=False, **kwargs):
        """Returns the most likely next state"""
        raise NotImplementedError
        return self.sample(state, action)

def get_model_actions():
    return MoveEast, MoveWest, MoveNorth, MoveSouth, LandAction()

def get_model_observations():
    return PlaneObservation(True), PlaneObservation(False) 

class PolicyModel(pomdp_py.RolloutPolicy):
    def __init__(self, n, k):
        # old: {"wait-wind", "change-airport", "land"}}
        self._move_actions = {MoveEast, MoveWest, MoveNorth, MoveSouth}
        self._other_actions = {LandAction()} # {WaitAction(), LandAction()}
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

            if state.position != "crashed" and state.position != "landed":
                plane_x, plane_y = state.coordinates
                if plane_x == self._k - 1:
                    motions.remove(MoveEast)
                if plane_y == 0:
                    motions.remove(MoveSouth)
                if plane_y == self._n - 1:
                    motions.remove(MoveNorth)
                if plane_x == 0:
                    motions.remove(MoveWest)  # TODO: correct?

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
        if isinstance(action, MoveAction) and (next_state.position != "landed") and (next_state.position != "crashed"): #Changed WaitAction to move
            if observation.wind == next_state.wind:
                return 1.0 - self.noise  # prob of getting correct wind
            else:
                return self.noise  # p of incorrect wind
        else:
            if observation.wind is None:
                return 1.0 - config.EPSILON  # expected to receive no observation
            else:
                return config.EPSILON

    def sample(self, next_state, action):
        if isinstance(action, MoveAction) and (next_state.position != "landed") and (next_state.position != "crashed"):
            thresh = 1.0 - self.noise

            if random.uniform(0, 1) < thresh:
                return PlaneObservation(next_state.wind)
            else:
                return PlaneObservation(not next_state.wind)
        else:
            return PlaneObservation(None)


class RewardModel(pomdp_py.RewardModel):
    def __init__(self):
        self._max_punish = -1000 #TODO: THIS SHOULD BE FROM CONFIG #is this needed?

    # TODO: make sure we reward if landed is next state
    def _reward_func(self, state, action):
        if state.position == "landed": #TODO: Enough time to give reward?
            return 100
        elif state.position == "crashed":
            return self._max_punish
        #elif isinstance(action, WaitAction):  # Small punishment for waiting
            #return -1
        elif isinstance(action, MoveAction):
            return -1  # no punishment for moving
        elif isinstance(action, LandAction):
            # or (state.coordinates == LINKOPING_LOCATION and state.wind == False):
            #if (state.coordinates == config.MALMEN_LOCATION) and (state.position == "landing"):
                #return 1000
            return -5  # punish for trying to land when not able to
        return 0

    def sample(self, state, action, next_state):
        # deterministic
        return self._reward_func(state, action)

    def max_punishment(self):
        return self._max_punish

    # how to return max reward for normalization?

