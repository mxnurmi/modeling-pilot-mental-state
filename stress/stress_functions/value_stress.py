import os
import sys
import inspect
import math

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import config

# Stress should come from either high uncertainty with likely negative reward or
# high chance of negative reward

def compute_value_stress(dd):
    #dd = TreeDebugger(agent.tree)
    expected_value = dd.c.value
    #print("value stress")
    # print(expected_value)
    # better way to adjust max punish?

    #maximum_mean_stress = -1 * (config.MAX_PUNISH / 2) # This is hacky way to find some threshold when we want to have about max stress. 
    #print("max mean stress")
    #print(maximum_mean_stress)
    input_value = expected_value/-250
    norm_value_tanh = math.tanh(input_value)
    norm_value_stress = max(0, norm_value_tanh) # ensure we dont go over 1

    #norm_value_stress = min(expected_value, 0) / (config.MAX_PUNISH - 10)

    return norm_value_stress