from pomdp_py.utils import TreeDebugger

import os
import sys
import inspect

# Same folder imports
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)

from attribute_stress import compute_attribute_stress
from predictability_controllability_novelty_stress import compute_pred_ctrl_nov_stress
from value_stress import compute_value_stress

# Stress: Stress is created by acting normally in extraordinary situations

def compute_stress(agent, num_sims):
    """

    num_sims: number of simulations done on this round. Used to normalize complexity. 
    """

    # Annoying to use treedebbuger, should have direct access NOTE: maybe implement later

    # BRIEF DOCUMENTATION:
    # print(dd.bestseqd(2)) # best sequence
    # print(dd.nn) # all nodes
    # print(dd.nv) # vnodes
    # print(dd.nq) # whst are qnodes -> below
    # print(dd.c.children) # children of a node
    # print(dd.p(2)) # print 2 layers of the tree
    # print(dd[0].value) # access value of specific node
    # print(dd.c.value) # of current node
    # node.num_visits # for visit amounts
    # node.value # for value

    # QNodes (value represents Q(b,a); children are observations) e.g. actions
    # VNodes (value represents V(b); children are actions). e.g. None, True, False
    # combining qnodes to vnodes = all nodes

    # print(dd.mbp) # best path
    # STRESS MODELS
    # MODEL1: dd.c.value normalized between 0-1
    # MODEL2: Gas and wind
    # MODEL3: Value estimate in combination with amount of nodes?
    # maybe Qnodes to Vnodes ratio as control? -> not good

    dd = TreeDebugger(agent.tree)

    pred_ctrl_nov_stress = compute_pred_ctrl_nov_stress(agent, dd, num_sims)
    attribute_stress = compute_attribute_stress(agent)
    value_stress = compute_value_stress(dd)

    # TODO: Can we somehow measure when a change to an uncertain state happens? E.g. fuel dump has 10% chance but when it happens its unlikely so we should have a pump in stress
    # On the other hand, the effeect should be negative for stress to increase? E.g. wind changing benefits the pilot so should not stress?

    return value_stress, attribute_stress, pred_ctrl_nov_stress
