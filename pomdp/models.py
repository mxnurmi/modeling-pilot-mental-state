
import pomdp_py
import random

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

class TransitionModel(pomdp_py.TransitionModel):

    def __init__(self, n, k):
        self._n = n
        self._k = k

    def probability(self, next_state, state, action):
        # TODO: we should have transition between differing wind states
        # In the tiger example there are always two states and opening a door will corrspond
        # to (state, action, next_state) which is 0.5 always as long as action is opening
        # In this transitionmodel we can just explicitly give out all of the probabilities instead

        if next_state != self.sample(state, action):
            return config.EPSILON
        else:
            return 1 - config.EPSILON

    def sample(self, state, action):
        # TODO: better structure
        if state.location == "landed" or state.location == "crashed":
            # nothing changes
            # return PlaneState(state.location, state.wind, state.fuel)
            return PlaneState(config.LINKOPING_LOCATION, True, config.START_FUEL)  # reset

        if state.fuel < 1:
            return PlaneState("crashed", True, state.fuel)


        # TODO: Wind should be estimated from previous wind
        wind_state = random.choices([True, False], weights=[
                                    config.WIND_PROB, 1-config.WIND_PROB], k=1)[0]

        # TODO: Should fuel dump amount be determined by config?s?
        fuel_state = random.choices([state.fuel - 1, state.fuel - 5], weights=[
                                    config.FUEL_PROB, 1-config.FUEL_PROB], k=1)[0]

        if isinstance(action, LandAction):
            if (state.location == config.MALMEN_LOCATION) or (state.location == config.LINKOPING_LOCATION and state.wind == False):
                return PlaneState("landed", True, state.fuel)
            else:
                return PlaneState(state.location, wind_state, state.fuel - 2)

        if isinstance(action, WaitAction):
            #windy_state = PlaneState(state.location, True, state.fuel - 1)
            return PlaneState(state.location, wind_state, fuel_state)
            # return random.choices([windy_state, non_windy_state], weights=[0.7, 0.3], k=1)[0]

        if isinstance(action, MoveAction):
            # We handle actions that go beyond the task domain with min + max

            new_location = (max(0, min(state.location[0] + action.motion[0], self._n - 1)),
                            max(0, min(state.location[1] + action.motion[1], self._k - 1)))

            return PlaneState(new_location, wind_state, fuel_state)

        # backup. TODO: Better solution?
        return PlaneState(state.location, state.wind, fuel_state)

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
                    motions.remove(MoveEast)
                if plane_y == 0:
                    motions.remove(MoveSouth)
                if plane_y == self._n - 1:
                    motions.remove(MoveNorth)
                if plane_x == 0:
                    motions.remove(MoveWest)  # TODO: correct?

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
                return 1.0 - config.EPSILON  # expected to receive no observation
            else:
                return config.EPSILON

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
            return -1
        elif isinstance(action, MoveAction):
            return -1  # no punishment for moving
        elif isinstance(action, LandAction):
            # or (state.location == LINKOPING_LOCATION and state.wind == False):
            if (state.location == config.MALMEN_LOCATION):
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

