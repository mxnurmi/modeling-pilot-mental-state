import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import config

def compute_value_stress(dd):
    #dd = TreeDebugger(agent.tree)
    expected_value = dd.c.value
    #print("value stress")
    # print(expected_value)
    # better way to adjust max punish?
    norm_value_stress = min(expected_value, 0) / (config.MAX_PUNISH - 10)
    # print(norm_value_stress)
    return norm_value_stress