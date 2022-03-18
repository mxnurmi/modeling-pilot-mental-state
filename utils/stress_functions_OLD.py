
from scipy.stats import norm
import math
import random
#import numpy as np

#We should estimate the wind based on previous wind state

# TODO: The wind is only relevant for stress near Linköping, 
# so we should focus on that!

# TODO: What generates stress?
# -> every turn wind doesn't change -> more stress
# -> every turn fuel gets lower -> more stress
# -> uncertainty -> more stress

def normalize_value(x, min_value, max_value):
    return ((x - min_value) / (max_value - min_value))


def stress_model(function_name, plane_state=None, start_fuel=None, current_wind=None, plane_problem=None):
    # TODO: Having these flexible none types is bad or they should be then checked by each stress function or something
    # wind = plane_state.wind
    # to normalize:
    # https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range

    if function_name == "purely_wind_based":

        # TODO: We need some kind of function that balances between 0 and 1 
        # where after 25 wind, we start a slow rice and after 45 will have stress of
        # 1

        return current_wind

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
        #print(expected)

        # wind limits control and amount of fuel increases control
        # how to add uncertainty?
        # lack of control and uncertainty could be by having it so
        # that the goals state is "out of reach". This would be
        # lack of control as agent is unable to directly get to goal

        #.plane_problem.env.state_transition(action, execute=True)
        # generate expected negative reward by sampling outcomes based on possible actions
        return 0
