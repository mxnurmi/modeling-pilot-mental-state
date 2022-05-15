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
    # complexity should be better normalized, but how? -> tanh
    # how well does the num_sims reflect to complexity??
    norm_complexity = math.tanh(complexity/num_sims)
    #norm_complexity = min(complexity/num_sims, 1)
    return norm_complexity

def compute_pred_ctrl_stress(agent, dd, num_sims):
    # Nodes could reflect uncertainty (amount of options) when combined with Entropy
    # Value could reflect controllability (lack of control is negative, having control positive)

    # get probabilities for each state
    #state_prob_dict = agent.cur_belief.get_histogram()
    #all_state_probs = []
    #for key in state_prob_dict:
    #    state = state_prob_dict[key]
    #    all_state_probs.append(state)

    #shannon_entropy = entropy(all_state_probs, base=2)
    # shannon entropy normalized between 0 and 1
    #norm_entropy = normalized_entropy(all_state_probs)
    # -> Higher entropy means that we dont know which event is likely to happen -> leads to stress
    # on the other hand, it should be noted that lower entropy means higher surprise!!
    norm_entropy = compute_predictability_stress(agent)

    complexity = dd.nn
    norm_complexity = min(complexity/num_sims, 1)

    # NOTE: Possible ways to normalize: Sigmoid etc?
    # vs. hearing
    # Humans contextualize emotions -> same for stress
    # Etsi: emotion regualization 
    # (esim. Edmund T Rolls, esim. brain and emotion)
    # Scherer (2009 -> coping ja hallinta) -- component process model!!! -> lisää knoppina mutta älä implementoi!
    # tilannetietoisuuden yhteys stressiin -> tilannetietoisuus esim entropyn kautta
    # tilannetietoisuus todnäk ratkaiseva pilottien kohdalla
    # artikkeleita esim. situational awarness and stress
    # Artikkeli "If it changes it must be a process"
    # if some problems are due to gridworld, do not try to solve them!
    # Seuraaviin askeliin: Siirtäminen continuous maailmaan
    # PPO -- softactor critic mahdollisia muit algoritmejä: dqn
    ## Mallilla mallinnetaan stressiä ja sitten katsotaan mitä käy ja voidaan sitten tarvittaessa ottaa pilotilta kontrollii
    # Onko lentomallissa jotain millä voidaan simuloida pilottikohtaisia tekijöitä. Esim kokemus

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

    # maybe use value nodes limited to specific depth split with total depth

    # maybe a lot of actions vs a few rewards == too complex? -> nope
    # TODO: it should be the amount of actions but how to normalize???

    #print("rewards/actions")
    #print(dd.nq / dd.nv)
    # TODO: Check which value goes over 1 here!

    pred_ctrl_stress = ( norm_complexity + (2 * norm_entropy) ) / 3

    #print("complexity")
    #print(norm_complexity)
    #print("predictability")
    #print(norm_entropy)
    #print("combined")
    #print(pred_ctrl_stress)

    return pred_ctrl_stress


