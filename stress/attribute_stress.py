import config
import os
import sys
import inspect

from scipy.stats import norm

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


def normalize_value(x, min_value, max_value):
    return ((x - min_value) / (max_value - min_value))


def compute_attribute_stress(agent):

    belief = agent.cur_belief.mpe()
    wind_state = belief.wind
    fuel_state = belief.fuel
    height_state = belief.position

    if height_state == "flying":
        start_fuel = config.START_FUEL
        x = norm.rvs(loc=35, scale=5, size=1)[0]
        maxx = 60
        minx = 25
        # limit min/max to 15/55 so we cant get values outside those and can normalize
        x = max(min(x, maxx), minx)
        # normalize between min and max stress
        x = normalize_value(x, minx, maxx)
        wind_estimation = x
        attribute_stress = (wind_estimation + (1 - fuel_state/start_fuel)) / 2
    else:
        # Scaled so that we are more likely to stress from wind
        start_fuel = config.START_FUEL
        x = norm.rvs(loc=35, scale=5, size=1)[0]
        maxx = 45
        minx = 10
        x = max(min(x, maxx), minx)
        x = normalize_value(x, minx, maxx)
        wind_estimation = x
        attribute_stress = (wind_estimation + (1 - fuel_state/start_fuel)) / 2


    
    return attribute_stress
