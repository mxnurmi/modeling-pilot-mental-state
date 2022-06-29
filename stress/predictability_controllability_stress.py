import math

#from scipy.stats import entropy

def normalized_entropy(probs):
    len_data = len(probs)
    base = 2.
    if len_data <= 1:
        return 0

    ent = 0
    for p in probs:
        if p > 0.:
            ent -= (p * math.log(p, base)) / math.log(len_data, base)

    return ent

def compute_predictability_stress(agent):

    state_prob_dict = agent.cur_belief.get_histogram()
    all_state_probs = []
    for key in state_prob_dict:
        state = state_prob_dict[key]
        all_state_probs.append(state)

    norm_entropy = normalized_entropy(all_state_probs)
    return norm_entropy


def compute_complexity_stress(dd, num_sims):
    complexity = dd.nn

    # how well does the num_sims reflect to complexity??
    norm_complexity = math.tanh(complexity/num_sims)

    return norm_complexity

def compute_pred_ctrl_stress(agent, dd, num_sims, wc=1, wp=2):

    # -> Higher entropy means that we dont know which event is likely to happen -> leads to stress
    # on the other hand, it should be noted that lower entropy means higher surprise!!
    norm_entropy = compute_predictability_stress(agent)

    complexity = dd.nn
    norm_complexity = math.tanh(complexity/num_sims)

    #print("complexity")
    #print(num_sims)
    #print("value")
    #print(dd.c.value)
    #print("all nodes")
    #print(dd.nn) # all nodes
    #print("value nodes")
    #print(dd.nv) # vnodes
    #print("action nodes")
    #print(dd.nq)

    # maybe use value nodes limited to specific depth split with total depth?

    pred_ctrl_stress = ( wc * norm_complexity + (wp * norm_entropy) ) / (wp + wc)

    # No way to add value into this function?
    # negative value should amplify, should we just use min() to ensure that the end result is below 1?

    return pred_ctrl_stress


