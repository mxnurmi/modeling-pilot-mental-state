
from scipy.stats import norm
import math
import random
#import numpy as np


def normalize_value(x, min_value, max_value):
    return ((x - min_value) / (max_value - min_value))

def stress_function(function_name, plane_state, start_fuel):

    #wind = plane_state.wind
    # to normalize:
    # https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
    fuel = plane_state.fuel

    if function_name == "sine_wind_based":
        x = random.randint(25, 45)  # wind speed
        wind_estimation = abs(math.sin(x))
        stress = (wind_estimation + (1 - fuel/start_fuel)) / 2

        return stress

    if function_name == "normal_wind_based":
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
        # generate expected negative reward by sampling outcomes based on possible actions
        return 0
