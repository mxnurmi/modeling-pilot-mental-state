import copy

from datetime import datetime, timedelta

import pyqtgraph as pg

from xplaneconnect import xpc

from stress import stress_estimator

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import geopandas as gpd
import numpy as np
import shapely

import pomdp_py

#import multiprocessing
from multiprocessing import Process, Value, Manager, Array

from pomdp.models import *
from pomdp.problem import PlaneProblem
from stress import stress_estimator

import config

# TODO:
# Wind level + Stress level

# TODO next:
# Fly and check that everything maps

# FIX THE CONFIG FILE THING!

# %%

plane_in_grid = []
airport1_in_grid = []
airport2_in_grid = []


def minimap(plane_coordinates_x, plane_coordinates_y, run, plane_in_grid, airport1_in_grid, airport2_in_grid):
    full_map = gpd.read_file(
        "./visualizers/alla_valdistrikt/alla_valdistrikt.shp")

    locations = ["Kärna", "S:t Lars", "Domkyrko \d* \(",
                 "Berga \d \(", "Slaka", "Ryd \d", "Skäggetorp", "Landeryd 1",
                 "Johannelund \d", "Landeryd \d"]

    map = full_map[full_map['VDNAMN'].str.contains('|'.join(locations))]
    map = map.to_crs('epsg:4326')

    def draw_map_with_locations(map, coordinates_airport1=None, coordinates_airport2=None):

        if coordinates_airport1 == None:
            x_ap1, y_ap1 = (15.670330652, 58.40499838)  # coordinates linköping
        else:
            x_ap1, y_ap1 = coordinates_airport1

        if coordinates_airport2 == None:
            x_ap2, y_ap2 = (15.51647, 58.41102)  # coordinates malmen
        else:
            x_ap2, y_ap2 = coordinates_airport2

        y_p = plane_coordinates_x.value
        x_p = plane_coordinates_y.value
        # print("coordinates:")
        #print(y_p, x_p)

        xmin, ymin, xmax, ymax = map.total_bounds
        # how many cells across and down
        n_cells = 20
        cell_size = (xmax-xmin) / n_cells

        grid_cells = []
        airport_cells = []
        plane_cells = []

        x_grid_number = 0
        y_grid_number = 0

        # There seems to be one extra in each grid square
        for x0 in np.arange(xmin, xmax+cell_size, cell_size):
            x_grid_number += 1
            y_grid_number = 0
            for y0 in np.arange(ymin, ymax+cell_size, cell_size):
                y_grid_number += 1

                x1 = x0-cell_size
                y1 = y0+cell_size

                grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))

                if (x_p > x1) and (x_p < x0) and (y_p > y0) and (y_p < y1):
                    plane_cells.append(shapely.geometry.box(x0, y0, x1, y1))

                elif (x_ap1 > x1) and (x_ap1 < x0) and (y_ap1 > y0) and (y_ap1 < y1):
                    airport_cells.append(shapely.geometry.box(x0, y0, x1, y1))

                elif (x_ap2 > x1) and (x_ap2 < x0) and (y_ap2 > y0) and (y_ap2 < y1):
                    airport_cells.append(shapely.geometry.box(x0, y0, x1, y1))

                # TODO: Annoying double if structure (redundant) as elif needed above?
                if (x_p > x1) and (x_p < x0) and (y_p > y0) and (y_p < y1):
                    plane_in_grid[0] = x_grid_number
                    plane_in_grid[1] = y_grid_number

                if (x_ap1 > x1) and (x_ap1 < x0) and (y_ap1 > y0) and (y_ap1 < y1):
                    airport1_in_grid[0] = x_grid_number
                    airport1_in_grid[1] = y_grid_number

                if (x_ap2 > x1) and (x_ap2 < x0) and (y_ap2 > y0) and (y_ap2 < y1):
                    airport2_in_grid[0] = x_grid_number
                    airport2_in_grid[1] = y_grid_number

        # TODO: IF airport 1 is under the agent it is not detected by the elif loop
        # -> CHANGE ELIFS TO ELSE SO THAT IT ALWAYS GOES THROUGH THAT
        return grid_cells, airport_cells, plane_cells

    fig, ax = plt.subplots(1, figsize=(5, 5))

    def animate(i):
        if not run.is_set():
            exit()

        grid_cells, airport_cells, plane_cells = draw_map_with_locations(map)

        ax2 = map.plot(markersize=.1, figsize=(5, 5),
                       column='VDNAMN', cmap='jet', ax=ax)
        plt.autoscale(False)

        cell = gpd.GeoDataFrame(grid_cells, columns=['geometry'])
        cell.plot(ax=ax2, facecolor="none", edgecolor='grey')

        airports = gpd.GeoDataFrame(airport_cells, columns=['geometry'])
        airports.plot(ax=ax2, facecolor="black", edgecolor='black')

        plane = gpd.GeoDataFrame(plane_cells, columns=['geometry'])
        plane.plot(ax=ax2, facecolor="white", edgecolor='black')

    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()


# TODO: We should make this so that instead of running agent directly, agent is ran on csv data and this is only used to gather all the parameters
def monitor(plane_coordinates_x, plane_coordinates_y, run, plane_in_grid, plot_wind=True):
    """
    params
    ======
        plane_coordinates_x: Tracks the planes real x-axis coordinates. Used to communicate them to minimap()
        plane_coordinates_y: Tracks the planes real y-axis coordinates. Used to communicate them to minimap()
        run: multiprocessing manager
        plane_in_grid: Location of plane mapped to grid coordinates
        plot_wind: Whether we plot separate window for wind
    
    """

    update_interval = 0.1  # seconds, originally 0.05 = 20 Hz

    app = pg.mkQApp("python xplane monitor")

    win = pg.GraphicsLayoutWidget(show=True)
    win.resize(500, 300)  # pixels
    win.setWindowTitle("XPlane system monitor")

    if plot_wind:
        p1 = win.addPlot(title="wind", row=0, col=0)
        p1.showGrid(y=True)

    p2 = win.addPlot(title="stress", row=0, col=1)
    p2.showGrid(y=True)

    plot_array_max_length = 300  # how many data points to hold in our arrays and graph

    x_axis_counters = []  # 0, 1, 2, 3, etc. just basic x-axis values used for plotting
    wind_history = []
    stress_history = []

    global last_update
    global i

    start = datetime.now()
    last_update = start
    i = 1  # initialize x_axis_counter

    # ----- AGENT initialization ------

    plane_status = "takeoff"
    wind_status = True
    init_true_state = PlaneState(
        (plane_in_grid[0], plane_in_grid[1]), plane_status, wind_status, 20)

    init_belief = pomdp_py.Particles([init_true_state])

    plane_problem = PlaneProblem(21, 10, init_true_state, init_belief)
    plane_problem.agent.set_belief(init_belief, prior=True)

    pomcp = pomdp_py.POMCP(max_depth=6, discount_factor=0.85,  # what does the discount_factor do?
                           planning_time=0.5, num_sims=-1, exploration_const=100,
                           rollout_policy=plane_problem.agent.policy_model,
                           show_progress=False, pbar_update_interval=1000)

    # ----- ----- ------

    last_stress_update = datetime.now()
    stress = 0

    with xpc.XPlaneConnect() as client:
        try:
            while True:
                if (datetime.now() > last_update + timedelta(milliseconds=update_interval * 1000)):
                    last_update = datetime.now()
                    posi = client.getPOSI()
                    ctrl = client.getCTRL()
                    pg.QtGui.QApplication.processEvents()

                    # print("Loc: (%4f, %4f, %4f) Aileron:%2f Elevator:%2f Rudder:%2f\n"\
                    # % (posi[0], posi[1], posi[2], ctrl[1], ctrl[0], ctrl[2]))

                    drefs = ["sim/weather/wind_speed_kt[0]",
                             "sim/weather/wind_speed_kt[1]", "sim/weather/wind_speed_kt[2]"]
                    #values = 30
                    #dref= "sim/cockpit/switches/gear_handle_status"
                    #dref = "sim/weather/wind_speed_kt[1]"
                    #client.sendDREF(dref, values)

                    # the height of the wind: wind_altitude_msl_m[0]
                    # the speed of the wind: wind_speed_kt[0]
                    # the direction of the wind: wind_direction_degt[0]

                    # TODO: add wind simulator and something that controls wind to system
                    # add fuel measurement

                    lat = posi[0]
                    longi = posi[1]
                    plane_coordinates_x.value = lat
                    plane_coordinates_y.value = longi

                    all_winds = client.getDREFs(drefs)
                    current_wind = all_winds[0][0]

                    fuel = client.getDREF("sim/cockpit2/fuel/fuel_quantity")
                    altitude = client.getDREF(
                        "sim/cockpit/pressure/cabin_altitude_actual_m_msl")
                    # fuel pressure per tank: sim/cockpit2/engine/indicators/fuel_pressure_psi
                    # acf_max_FUELP max fuel pressure
                    # fuel_quantity

                    # Stress computations
                    action = pomcp.plan(plane_problem.agent)

                    env_reward = plane_problem.env.state_transition(
                        action, execute=False)
                        # TODO: we should only execute when we truly move to next state

                    # TODO: Is this correct though?
                    real_observation = plane_problem.env.provide_observation(
                        plane_problem.agent.observation_model, action)
                    # TODO: history should only be updated when we move to next state and the actiuon should be pilot controlled not the one from agent!
                    plane_problem.agent.update_history( 
                        action, real_observation)
                    pomcp.update(plane_problem.agent, action, real_observation)
                    agent_state = copy.deepcopy(plane_problem.env.state)

                    value_stress, attribute_stress, predict_control_stress, ctrl_stress, pred_stress = stress_estimator.compute_stress(
                        plane_problem.agent, num_sims=pomcp.last_num_sims) # TODO: Maybe return dict instead?

                    #fuel = client.sendDREF("sim/cockpit2/fuel/fuel_quantity", values=[(0.0,0.0,0,0,0,0,0,0,0)]) -> doesnt work

                    if altitude[0] > 400:
                        plane_status = "flying"

                    wind_noise = random.uniform(-3.0, 3.0)

                    if altitude[0] > 1000:
                        set_drefs = ["sim/weather/wind_speed_kt[0]", "sim/weather/wind_direction_degt[0]", "wind_altitude_msl_m[0]"]
                        wind_speed = max(0, 0 + wind_noise)
                        values = [wind_speed, 100, 500]
                        client.sendDREFs(set_drefs, values)
                        wind_status = False
                    #else:
                        #set_drefs = ["sim/weather/wind_speed_kt[0]", "sim/weather/wind_direction_degt[0]", "wind_altitude_msl_m[0]"]
                        #wind_speed = 30 + wind_noise
                        #values = [wind_speed, 100, 300]
                        #client.sendDREFs(set_drefs, values)

                    # only update stress every five seconds
                    if (datetime.now() > last_stress_update + timedelta(milliseconds=update_interval * 18000)):
                        last_stress_update = datetime.now()
                        stress = predict_control_stress

                    # TODO: We should somehow handle landing vs takeoff

                    # TODO: We should only reset when the state does not change and even then the history should be maintained
                    init_true_state = PlaneState(
                        (plane_in_grid[0], plane_in_grid[1]), plane_status, wind_status, 20)

                    init_belief = pomdp_py.Particles([init_true_state])

                    plane_problem = PlaneProblem(
                        21, 10, init_true_state, init_belief)
                    plane_problem.agent.set_belief(init_belief, prior=True)

                    # print("grid")
                    #print("plane:", plane_in_grid[:], "linkoping:", airport1_in_grid[:], "malmen:", airport2_in_grid[:])

                    # boolean check to make sure we limit the plot size to window
                    if(len(x_axis_counters) > plot_array_max_length):
                        x_axis_counters.pop(0)
                        wind_history.pop(0)
                        stress_history.pop(0)

                        x_axis_counters.append(i)
                        wind_history.append(current_wind)
                        stress_history.append(stress)
                    else:
                        x_axis_counters.append(i)
                        wind_history.append(current_wind)
                        stress_history.append(stress)
                    i = i + 1

                    if plot_wind:
                        p1.plot(x_axis_counters, wind_history, pen=0, clear=True)
                    p2.plot(x_axis_counters, stress_history, pen=0, clear=True)

        except Exception as e:
            run.clear()
            raise


def run_simulator_tools():
    plane_coordinates_x = Value('d', 15.670330652)
    plane_coordinates_y = Value('d', 58.40499838)

    plane_in_grid = Array('i', [14, 6])
    airport1_in_grid = Array('i', [14, 6])
    airport2_in_grid = Array('i', [5, 6])
    plot_wind = True

    #config.init_scenario(wind=1, fuel_amount=11, fuel_keep_chance=1, n=(21, 10), airport1_coor=(14,6), airport2_coor=(5,6))

    manager = Manager()
    run = manager.Event()
    run.set()  # We should keep running.

    p1 = Process(target=minimap, args=(
        plane_coordinates_x, plane_coordinates_y, run, plane_in_grid, airport1_in_grid, airport2_in_grid))
    p2 = Process(target=monitor, args=(
        plane_coordinates_x, plane_coordinates_y, run, plane_in_grid, plot_wind))
    # TODO: One of the args should be the agent

    p1.start()
    p2.start()

    p1.join()
    p2.join()


if __name__ == "__main__":
    run_simulator_tools()


# %%
