
from scipy.stats import norm
import math
import random
#import numpy as np

#We should estimate the wind based on previous wind state

# TODO: The wind is only relevant for stress near Linköping, 
# so we should focus on that!

def normalize_value(x, min_value, max_value):
    return ((x - min_value) / (max_value - min_value))

def stress_function(function_name, plane_state=None, start_fuel=None, plane_problem=None):

    #wind = plane_state.wind
    # to normalize:
    # https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
    

    if function_name == "sine_wind_based":
        # this should be fixed so that when wind is high -> expect lower
        # whin wind is low -> expect higher
        fuel = plane_state.fuel
        x = random.randint(25, 45)  # wind speed
        wind_estimation = abs(math.sin(x))
        stress = (wind_estimation + (1 - fuel/start_fuel)) / 2

        return stress

    if function_name == "normal_wind_based":
        fuel = plane_state.fuel
        x = norm.rvs(loc=35, scale=5, size=1)
        maxx = 55
        minx = 15
        # limit min/max to 15/55 so we cant get values outside those and can normalize
        x = max(min(x, maxx), minx)
        # normalize between min and max stress
        x = normalize_value(x, minx, maxx)
        wind_estimation = x
        stress = (wind_estimation + (1 - fuel/start_fuel)) / 2
        return stress

    if function_name == "expected_negative_reward":

        actions = plane_problem.agent.policy_model.get_all_actions()
        expected = 0
        # TODO: how to compute uncertainty between states? should we use the action being taken?
        # maybe if after two steps we have actions which all have expected negative outcome
        # maybe given an action we sample to depth of 3 and count what is the expected punishment?

        for action in actions:
            next_state, value = plane_problem.env.state_transition(action, execute=False) 
            expected += value
        print(expected)

        #.plane_problem.env.state_transition(action, execute=True)
        # generate expected negative reward by sampling outcomes based on possible actions
        return 0
