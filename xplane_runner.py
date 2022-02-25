from datetime import datetime, timedelta

import pyqtgraph as pg

from xplaneconnect import xpc
from utils.stress_functions import stress_model

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import geopandas as gpd
import numpy as np
import shapely

#import multiprocessing
from multiprocessing import Process, Value

# TODO:
# Wind level + Stress level

# TODO next:
# Fly and check that everything maps

# %%

def minimap(plane_coordinates_x, plane_coordinates_y):
    full_map = gpd.read_file("./visualizers/alla_valdistrikt/alla_valdistrikt.shp")

    locations = ["Kärna", "S:t Lars", "Domkyrko \d* \(",
                "Berga \d \(", "Slaka", "Ryd \d", "Skäggetorp", "Landeryd 1",
                "Johannelund \d", "Landeryd \d"]

    map = full_map[full_map['VDNAMN'].str.contains('|'.join(locations))]
    map = map.to_crs('epsg:4326')

    def draw_map_with_locations(map, coordinates_airport1=None, coordinates_airport2=None):

        if coordinates_airport1 == None:
            x_ap1, y_ap1 = (15.670330652, 58.40499838) #coordinates linköping
        else:
            x_ap1, y_ap1 = coordinates_airport1

        if coordinates_airport2 == None:
            x_ap2, y_ap2 = (15.51647, 58.41102) #coordinates malmen
        else:
            x_ap2, y_ap2 = coordinates_airport2

        y_p = plane_coordinates_x.value
        x_p = plane_coordinates_y.value
        #print("coordinates:")
        #print(y_p, x_p)
        
        xmin, ymin, xmax, ymax = map.total_bounds
        # how many cells across and down
        n_cells = 30
        cell_size = (xmax-xmin) / n_cells
        
        grid_cells = []
        airport_cells = []
        plane_cells = []

        for x0 in np.arange(xmin, xmax+cell_size, cell_size):
            for y0 in np.arange(ymin, ymax+cell_size, cell_size):
                x1 = x0-cell_size
                y1 = y0+cell_size

                grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))

                if (x_p > x1) and (x_p < x0) and (y_p > y0) and (y_p < y1):
                    plane_cells.append(shapely.geometry.box(x0, y0, x1, y1))

                elif (x_ap1 > x1) and (x_ap1 < x0) and (y_ap1 > y0) and (y_ap1 < y1):
                    airport_cells.append(shapely.geometry.box(x0, y0, x1, y1))

                elif (x_ap2 > x1) and (x_ap2 < x0) and (y_ap2 > y0) and (y_ap2 < y1):
                    airport_cells.append(shapely.geometry.box(x0, y0, x1, y1))
                
        return grid_cells, airport_cells, plane_cells


    # %%

    fig, ax = plt.subplots(1, figsize=(5,5))

    def animate(i):
        grid_cells, airport_cells, plane_cells = draw_map_with_locations(map)

        ax2 = map.plot(markersize=.1, figsize=(5, 5), column='VDNAMN', cmap='jet', ax=ax)
        plt.autoscale(False)

        cell = gpd.GeoDataFrame(grid_cells, columns=['geometry'])
        cell.plot(ax=ax2, facecolor="none", edgecolor='grey')

        airports = gpd.GeoDataFrame(airport_cells, columns=['geometry'])
        airports.plot(ax=ax2, facecolor="black", edgecolor='black')

        plane = gpd.GeoDataFrame(plane_cells, columns=['geometry'])
        plane.plot(ax=ax2, facecolor="white", edgecolor='black')

    ani = animation.FuncAnimation(fig, animate, interval=1000) 
    plt.show()


def monitor(plane_coordinates_x, plane_coordinates_y):

    update_interval = 0.050 # seconds, 0.05 = 20 Hz

    app = pg.mkQApp("python xplane monitor")

    win = pg.GraphicsLayoutWidget(show=True)
    win.resize(500, 300) #pixels
    win.setWindowTitle("XPlane system monitor")

    p1 = win.addPlot(title="wind",row=0,col=0)
    p1.showGrid(y=True)

    p2 = win.addPlot(title="stress",row=0,col=1)
    p2.showGrid(y=True)

    plot_array_max_length = 300 # how many data points to hold in our arrays and graph

    x_axis_counters = [] #0, 1, 2, 3, etc. just basic x-axis values used for plotting
    wind_history = []
    stress_history = []

    global last_update
    global i

    start = datetime.now()
    last_update = start
    i = 1 # initialize x_axis_counter

    with xpc.XPlaneConnect() as client:
        while True:
            if (datetime.now() > last_update + timedelta(milliseconds = update_interval * 1000)):
                last_update = datetime.now()
                posi = client.getPOSI();
                ctrl = client.getCTRL();
                pg.QtGui.QApplication.processEvents()

                #print("Loc: (%4f, %4f, %4f) Aileron:%2f Elevator:%2f Rudder:%2f\n"\
                #% (posi[0], posi[1], posi[2], ctrl[1], ctrl[0], ctrl[2]))
                
                drefs = ["sim/weather/wind_speed_kt[0]", "sim/weather/wind_speed_kt[1]", "sim/weather/wind_speed_kt[2]"]
                #values = 30
                #dref= "sim/cockpit/switches/gear_handle_status"
                #dref = "sim/weather/wind_speed_kt[1]"
                #client.sendDREF(dref, values)

                lat = posi[0]
                longi = posi[1]
                plane_coordinates_x.value = lat
                plane_coordinates_y.value = longi

                all_winds = client.getDREFs(drefs)
                current_wind = all_winds[0][0]

                fuel = client.getDREF("sim/cockpit2/fuel/fuel_quantity")
                #fuel pressure per tank: sim/cockpit2/engine/indicators/fuel_pressure_psi
                #print("fuel for each tank")
                #print(fuel)

                # acf_max_FUELP max fuel pressure
                # fuel_quantity

                stress = stress_model("purely_wind_based", current_wind=current_wind)

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

                p1.plot(x_axis_counters, wind_history, pen=0, clear=True)
                p2.plot(x_axis_counters, stress_history, pen=0, clear=True)


if __name__ == "__main__":
    plane_coordinates_x = Value('d', 15.670330652)
    plane_coordinates_y = Value('d', 58.40499838)

    p1 = Process(target=minimap, args=(plane_coordinates_x, plane_coordinates_y))
    p2 = Process(target=monitor, args=(plane_coordinates_x, plane_coordinates_y))

    p1.start()
    p2.start()

    p1.join()
    p2.join()


# %%


# %%
