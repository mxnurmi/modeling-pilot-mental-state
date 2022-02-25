# %%

import pomdp_py
import copy
import random

from math import dist

from scipy.stats import entropy

#import pickle
from pomdp_py.utils import TreeDebugger

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

from visualizers.visual import get_coordinates
from utils.stress_functions import stress_model

from pomdp.models import *
from pomdp.problem import PlaneProblem

import config

# - Pickle support?

# TODO: Need a way to define:
# - Bunch of models
# - Bunch of training scenarios that can be each input to each model
# - Bunch of actual scenarios that can be each input to each model

# NOTE:
# We could approximate stress by keeping counts on how often the agent falls and if a given state
# lead to the agent falling. The higher the chance that it did, the more stressed agent is.
# However, this would approximate the threat to agent but how do we quantify uncertainty


def generate_random_state():
    location = config.LINKOPING_LOCATION
    return PlaneState(location, True, config.START_FUEL)


def generate_init_belief(num_particles):
    particles = []
    for _ in range(num_particles):
        particles.append(generate_random_state())

    return pomdp_py.Particles(particles)


class PlaneScenario():
    """
    Define the plane scenarios

    We have X options (that can happen concurrently):
    - Wind changes randomly near closest airport to make it unfavorable, but is favorable at second
    - 

    Parameters should probably be something like 
    crosswind at airport1 and crosswind at airport2?

    For some of these we need to calculate the square that is half
    Way to the destination
    """

    # Wind attributes:
    # Wind direction: wind_direction_degt[0] (north being 0)
    # Wind altitude: wind_altitude_msl_m[0]
    # Wind speed: wind_speed_kt[0]

    # Fuel visualizer?:
    # fuel_pressure_psi ?
    # rel_g_fuel ?
    # rel_fp_ind_0 ?

    # dump_fuel -> maybe a switch that dumps the fuel

    def __init__(self):
        wind_attributes = (0, 0, 0)  # dir, alt, speed
        fuel_dump = False  # will fuel be dumped mid flight
        turning_wind = False  # will wind turn mid flight
        refuse_landing = False  # the destination airport will refuse initial landing
        hidden_fuel = False  # fuel meter gets frozen

# wind should linearly transition to next average. Possibly we could have some normal sampling from that linear function as well
# gusts last max 20second and should only happen with maybe 1/8 or 1/10 of a chance


def print_status(frame, planner, plane_problem, action, env_reward, output=True):

    true_state = copy.deepcopy(plane_problem.env.state)

    print("==== Step %d ====" % (frame+1))
    print("True state: %s" % true_state)
    print("Belief: %s" % str(plane_problem.agent.cur_belief))
    print("Action: %s" % str(action))
    print("Reward: %s" % str(env_reward))
    print("Total reward: " + str(total_reward))
    if isinstance(planner, pomdp_py.POUCT):
        print("__num_sims__: %d" % planner.last_num_sims)
        print("__plan_time__: %.5f" % planner.last_planning_time)
    if isinstance(planner, pomdp_py.PORollout):
        print("__best_reward__: %d" % planner.last_best_reward)
    print("\n")


def runner_no_a(plane_problem, planner, nsteps=20, debug_tree=False, size=None, plot=False):
    """
    Runs the action-feedback loop of Plane problem POMDP

    Args:
        plane_problem (PlaneProblem): an instance of the plane problem.
        planner (Planner): a planner
        nsteps (int): Maximum number of steps to run this loop.
    """
    # ----Scenario task----
    # TODO: We should import the scenarios!

    total_reward = 0
    stress_states_sine = []
    stress_states_normal = []
    i = 0

    while i < nsteps:
        true_state = copy.deepcopy(plane_problem.env.state)
        true_location = true_state.location

        action = planner.plan(plane_problem.agent)
        env_reward = plane_problem.env.state_transition(
            action, execute=True)  # TODO: use this for sampling
        total_reward += env_reward

        print_status(i, planner, plane_problem, action, env_reward)

        if debug_tree == True:
            dd = TreeDebugger(plane_problem.agent.tree)
            print(dd.pp)
            print(dd.mbp)  # TODO: what is the "none" leaf?
            print("\n")

        real_observation = plane_problem.env.provide_observation(
            plane_problem.agent.observation_model, action)
        plane_problem.agent.update_history(action, real_observation)
        planner.update(plane_problem.agent, action, real_observation)

        # TODO: move this inside the stress function
        # TODO: Make sure we always pick strongest belief (now pick first?)
        for belief in plane_problem.agent.cur_belief:
            belief_state = belief
            break

        stress_sine = stress_model(
            "sine_wind_based", belief_state, start_fuel=config.START_FUEL)
        stress_normal = stress_model(
            "normal_wind_based", belief_state, start_fuel=config.START_FUEL)
        #stress_expected_reward = stress_function("expected_negative_reward", plane_problem=plane_problem)
        stress_states_sine.append(stress_sine)
        stress_states_normal.append(stress_normal)

        i += 1

    som = plane_problem.agent
    # with open(f'test.pickle', 'wb') as file:
    #pickle.dump(som, file)

    # load it
    # with open(f'test.pickle', 'rb') as file2:
    #s1_new = pickle.load(file2)

    # print("pickle")
    # print(s1_new)

    print("\nSine stress")
    print(stress_states_sine)

    print("\nNormal stress")
    print(stress_states_normal)

    print("\n")
    print("=== DONE ===")


def init_figure(coordinates, true_state):
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    fig.set_dpi(120)
    fig.set_size_inches(13, 8, forward=True)
    # bbox = dict(facecolor = 'grey', alpha = 0.5))
    plot_text = ax2.text(-0.8, -3.2, '', fontsize=15, weight="bold")
    im = ax2.imshow(coordinates, origin='lower', cmap='gray')
    ax2.set_title(f"Step 0", fontsize=20)
    plot_text.set_text("Action: -"
                       + "\n" + "Reward: -"
                       + "\n" + "Total reward: 0"
                       + "\n" + "State:" + str(true_state)
                       )
    ax1.set_title(f"Stress", fontsize=20)

    return fig, im, plot_text, ax1, ax2


def pomdp_step(plane_problem, planner):
    true_state = copy.deepcopy(plane_problem.env.state)
    action = planner.plan(plane_problem.agent)
    env_reward = plane_problem.env.state_transition(
        action, execute=True)  # TODO: use this for sampling
    real_observation = plane_problem.env.provide_observation(
        plane_problem.agent.observation_model, action)
    plane_problem.agent.update_history(action, real_observation)
    planner.update(plane_problem.agent, action, real_observation)
    plane_location = true_state.location

    return action, true_state, plane_location, env_reward


total_reward = 0


def compute_stress(agent):
    # Annoying to use treedebbuger, should have direct access NOTE: maybe implement later
    dd = TreeDebugger(agent.tree)

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

    # Nodes could reflect uncertainty (amount of options) when combined with Entropy
    # Value could reflect controllability (lack of control is negative, having control positive)

    # We should pick the optimal belief like this and not with the loop:
    print(agent.cur_belief.mpe())

    # so this is wrong:
    for belief in agent.cur_belief:
        belief_state = belief
        break

    # TODO: TEST THAT ABOVE LOOP IS INDEED SAME AS WITH THE ONE FROM PRINT

    # get probabilities for each state
    state_prob_dict = agent.cur_belief.get_histogram()
    all_state_probs = []
    for key in state_prob_dict:
        state = state_prob_dict[key]
        all_state_probs.append(state)

    # TODO: figure out what entropy means here: watch that video again:
    surprise = entropy(all_state_probs, base=2)

    complexity = dd.nn
    expected_value = dd.c.value

    print(agent.cur_belief.mpe())

    # TODO: compute stress using surprise, complexity, expected value here

    stress = 0
    return stress


def runner_a(plane_problem, planner, nsteps=20, size=None, save_animation=False, scenario_parameters=None):
    """
    Animates and runs the action-feedback loop of Plane problem POMDP

    Args:
        plane_problem (PlaneProblem): an instance of the plane problem.
        planner (Planner): a planner
        nsteps (int): Maximum number of steps to run this loop.
    """

    n, k = size  # TODO: size gets none by default, fix or do error handling
    width = n
    height = k
    true_state = copy.deepcopy(plane_problem.env.state)
    plane_location = true_state.location
    airport_location1 = config.MALMEN_LOCATION
    airport_location2 = config.LINKOPING_LOCATION

    coordinates = get_coordinates(
        width, height, plane_location, airport_location1, airport_location2)
    fig, im, plot_text, ax1, ax2 = init_figure(coordinates, true_state)

    def init_anim():
        coordinates = get_coordinates(
            width, height, plane_location, airport_location1, airport_location2)
        im.set_array(coordinates)
        return im

    stress_data = []
    frame_data = []

    def animate_func(frame):
        global total_reward  # hacky way to update total_reward

        if frame == nsteps-1:
            true_state = copy.deepcopy(plane_problem.env.state)
            plot_text.set_text("SIMULATION COMPLETE"
                               + "\nFinal state:" + str(true_state)
                               + "\nTotal reward:" + str(total_reward)
                               )

        else:
            action, true_state, plane_location, env_reward = pomdp_step(
                plane_problem, planner)

            coordinates = get_coordinates(
                width, height, plane_location, airport_location1, airport_location2)
            im.set_array(coordinates)

            cum_rewd = total_reward + env_reward
            total_reward = cum_rewd

            plot_text.set_text("Action: " + str(action)
                               + "\n" + "Reward: " + str(env_reward)
                               + "\n" + "Total reward: " + str(total_reward)
                               + "\n" + "State: " + str(true_state)
                               )

            ax2.set_title(f"Step {frame+1}", fontsize=20)

            print_status(frame, planner, plane_problem, action, env_reward)

            # TODO: Change colormap upon plane crash?

            compute_stress(plane_problem.agent)
            # stress_normal = stress_model(
            # "normal_wind_based", belief_state, start_fuel=config.START_FUEL)
            # stress_data.append(stress_normal[0])

            frame_data.append(frame)
            ax1.plot(frame_data, stress_data, color="r", lw=4)
            return im

    anim = animation.FuncAnimation(
        fig,
        animate_func,
        init_func=init_anim,
        frames=nsteps,
        interval=510,  # in ms
        repeat=False
    )
    if save_animation:
        anim.save("animation.gif", dpi=300, writer=PillowWriter(fps=2))
    else:
        plt.show()


def main(plot, steps=15, save_animation=False):

    # TODO: Init scenario here

    config.init_scenario(wind=1)  # this should be called in config!!

    init_true_state = PlaneState(
        config.LINKOPING_LOCATION, True, config.START_FUEL)
    init_belief = generate_init_belief(50)

    n = config.SIZE[0]
    k = config.SIZE[1]

    plane_problem = PlaneProblem(n, k, init_true_state, init_belief)

    plane_problem.agent.set_belief(init_belief, prior=True)

    pomcp = pomdp_py.POMCP(max_depth=5, discount_factor=0.85,  # what does the discount_factor do?
                           planning_time=2, num_sims=-1, exploration_const=100,
                           rollout_policy=plane_problem.agent.policy_model,
                           show_progress=False, pbar_update_interval=1000)

    if plot == False:
        runner_no_a(plane_problem, planner=pomcp,
                    nsteps=steps, size=(n, k))
    else:
        runner_a(plane_problem, planner=pomcp,
                 nsteps=steps, size=(n, k), save_animation=save_animation)


if __name__ == '__main__':
    main(plot=True, steps=15, save_animation=True)

# Stress should come from either high uncertainty with likely negative reward or
# high chance of negative reward

#### POUCT MODEL #####

    #po_rollout = pomdp_py.PORollout()

    # pouct = pomdp_py.POUCT(max_depth=10, discount_factor=0.999,  # what does the discount_factor do?
    #                       num_sims=1000, exploration_const=1000,
    #                       rollout_policy=plane_problem.agent.policy_model,
    #                       show_progress=True, pbar_update_interval=1000)

# %%
