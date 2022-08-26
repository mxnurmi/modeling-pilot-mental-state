import copy
from tkinter import N

from models import *
from problem import PlaneProblem
#from stress import stress_estimator

def generate_random_state(start_state):
    location = config.LINKOPING_LOCATION
    return PlaneState(location, start_state, True, config.START_FUEL)


def generate_init_belief(num_particles, start_state):
    particles = []
    for _ in range(num_particles):
        particles.append(generate_random_state(start_state))

    return pomdp_py.Particles(particles)

class RLAgentWrapper():
    def __init__(self, init_plane_in_grid, init_plane_state, init_wind_state, init_fuel_state):
        self.init_plane_in_grid = init_plane_in_grid # plane's initial location (List)
        self.init_plane_state = init_plane_state #"takeoff"
        self.init_wind_state = init_wind_state #True
        self.init_fuel_state = init_fuel_state #20

    def init_plane_problem(self):
        init_true_state = PlaneState(
            (self.init_plane_in_grid[0], self.init_plane_in_grid[1]), self.init_plane_state, self.init_wind_state, self.init_fuel_state)

        #init_belief = pomdp_py.Particles([init_true_state])
        init_belief = generate_init_belief(50, "pre-flight")
        self.plane_problem = PlaneProblem(self.init_plane_in_grid[0], self.init_plane_in_grid[1], init_true_state, init_belief)
        self.plane_problem.agent.set_belief(init_belief, prior=True)

        # TODO: Should these be defined somewhere separately?
        self.pomcp = pomdp_py.POMCP(max_depth=6, discount_factor=0.85, # what does the discount_factor do?
                                    planning_time=0.5, num_sims=-1, exploration_const=100,
                                    rollout_policy=self.plane_problem.agent.policy_model,
                                    show_progress=False, pbar_update_interval=1000)

    # TODO: Should have functions for either:
    # A) Computing predicted state but reseting agent's state in the beginning -> cant these be just be done after action??
    # B) Computing predicted state but maintaing agent's previous state in the beginning
    # BONUS FACTOR: We also need to know whether to use observation to update agent's state or
    # whther we should just reset agent's state so that it corresponds with the true state

    def find_new_state_no_ext_params(self):
        """ Computes POMCP-based best action and and uses it to update agent's state
        """

        action = self.pomcp.plan(self.plane_problem.agent)
        env_reward = self.plane_problem.env.state_transition(
            action, execute=True)
        real_observation = self.plane_problem.env.provide_observation(
            self.plane_problem.agent.observation_model, action)

        self.plane_problem.agent.update_history(action, 
                                                real_observation)
        self.pomcp.update(self.plane_problem.agent, 
                          action, 
                          real_observation)

        true_state = copy.deepcopy(self.plane_problem.env.state)
        #belief_atm = (self.plane_problem.agent.cur_belief)
        #print(belief_atm)
        plane_location = true_state.coordinates

        return action, true_state, plane_location, env_reward

    def find_new_state_with_ext_params(self):
        """ Computes new state, but only by using external parameters and by not actually executing the action 
            (TODO: external parameters might need to be used only to provide observation?)
        """

        action = self.pomcp.plan(self.plane_problem.agent)

        env_reward = self.plane_problem.env.state_transition(
            action, execute=False)
            # TODO: we should execute when we truly move to next state (coordinates of plane change)
            # The challenge is now that we cant know how the agent moves as the state transitions are not deterministic

        # TODO: This should be based on the real plane's state!
        real_observation = self.plane_problem.env.provide_observation(
            self.plane_problem.agent.observation_model, action)

        # TODO: history should only be updated when we move to next state and the action should be pilot controlled not the one from agent!
        self.plane_problem.agent.update_history( 
            action, real_observation)
        self.pomcp.update(self.plane_problem.agent, action, real_observation)
        agent_state = copy.deepcopy(self.plane_problem.env.state)

        #print(agent_state)
        plane_location = agent_state.coordinates

    def compute_stress(self, stress_estimator):
        value_stress, attribute_stress, predict_control_stress, ctrl_stress, pred_stress = stress_estimator.compute_stress(
                    self.plane_problem.agent, num_sims=self.pomcp.last_num_sims)

        return value_stress, attribute_stress, predict_control_stress, ctrl_stress, pred_stress

    def return_policy(self):
        return self.plane_problem.agent.policy_model

    def return_plane_position(self):
        return self.plane_problem.env.state.position

    def return_agent(self):
        """ Returns the actual, problem-bound agent
        """
        return self.plane_problem.agent

    def return_num_of_planning_sims(self):
        return self.pomcp.last_num_sims

    def return_plane_problem(self):
        return self.plane_problem

    def return_planner(self):
        return self.pomcp
    
    