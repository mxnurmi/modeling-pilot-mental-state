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

def compute_pred_ctrl_nov_stress(agent, dd, num_sims):
    # Nodes could reflect uncertainty (amount of options) when combined with Entropy
    # Value could reflect controllability (lack of control is negative, having control positive)

    # get probabilities for each state
    state_prob_dict = agent.cur_belief.get_histogram()
    all_state_probs = []
    for key in state_prob_dict:
        state = state_prob_dict[key]
        all_state_probs.append(state)

    #shannon_entropy = entropy(all_state_probs, base=2)
    # shannon entropy normalized between 0 and 1
    norm_entropy = normalized_entropy(all_state_probs)
    # -> Higher entropy means that we dont know which event is likely to happen -> leads to stress
    # on the other hand, it should be noted that lower entropy means higher surprise!!

    # complexity should be normalized, but how?
    complexity = dd.nn
    # print("complexity")
    # print(complexity)
    #print("normalized complexity")
    norm_complexity = complexity/num_sims

    pred_ctrl_stress = (norm_complexity + norm_entropy) / 2
    return pred_ctrl_stress


