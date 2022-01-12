from datetime import datetime, timedelta
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.widgets import MatplotlibWidget
from xplaneconnect import xpc
from PyQt6 import QtWidgets

import random

# TODO: We want to plot three things:
# Wind level
# Stress level
# Plane location on grid

update_interval = 0.050 # seconds, 0.05 = 20 Hz
start = datetime.now()
last_update = start

app = pg.mkQApp("python xplane monitor")

win = pg.GraphicsLayoutWidget(show=True)
win.resize(1000,600) #pixels
win.setWindowTitle("XPlane system monitor")

p1 = win.addPlot(title="wind",row=0,col=0)
p1.showGrid(y=True)

#p2 = win.addPlot(title="test",row=0,col=1)

#x = [1, 2, 3, 4]
#y = [1, 2, 3, 4]
#mw = MatplotlibWidget.MatplotlibWidget()
#subplot = mw.getFigure().add_subplot(111)
#subplot.plot(x,y)
#mw.draw()
#print("subplot:", type(mw))
#p2.addItem(subplot)


x_axis_counters = [] #0, 1, 2, 3, etc. just basic x-axis values used for plotting
wind_history = []

plot_array_max_length = 300 # how many data points to hold in our arrays and graph
i = 1 # initialize x_axis_counter

def monitor():
    global last_update
    global i

    with xpc.XPlaneConnect() as client:
        while True:
            if (datetime.now() > last_update + timedelta(milliseconds = update_interval * 1000)):
                last_update = datetime.now()
                posi = client.getPOSI();
                ctrl = client.getCTRL();
                pg.QtGui.QApplication.processEvents()

                print("Loc: (%4f, %4f, %4f) Aileron:%2f Elevator:%2f Rudder:%2f\n"\
                % (posi[0], posi[1], posi[2], ctrl[1], ctrl[0], ctrl[2]))
                
                dref = "sim/weather/wind_speed_kt[1]"
                values = 30
                #dref= "sim/cockpit/switches/gear_handle_status"
                #client.sendDREF(dref, values)
                current_wind = client.getDREF(dref)[0]

                # boolean check to make sure we limit the plot size to window
                if(len(x_axis_counters) > plot_array_max_length):
                    x_axis_counters.pop(0)
                    wind_history.pop(0)

                    x_axis_counters.append(i)
                    wind_history.append(current_wind)
                else:
                    x_axis_counters.append(i)
                    wind_history.append(current_wind)
                i = i + 1

                p1.plot(x_axis_counters, wind_history, pen=0, clear=True)


#importing libraries
# %%
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import geopandas as gpd
import numpy as np
import shapely

full_map = gpd.read_file("./alla_valdistrikt/alla_valdistrikt.shp")
#fig, ax = plt.subplots(figsize=(50, 50))

locations = ["Kärna", "S:t Lars", "Domkyrko \d* \(",
             "Berga \d \(", "Slaka", "Ryd \d", "Skäggetorp", "Landeryd 1",
             "Johannelund \d", "Landeryd \d"]

map = full_map[full_map['VDNAMN'].str.contains('|'.join(locations))]
map = map.to_crs('epsg:4326')

def draw_map_with_locations(map, coordinates_airplane, coordinates_airport1=None, coordinates_airport2=None):
        
    #sa_x = 15.670330652 #linköping
    #sa_y = 58.40499838

    #sa2_x = 15.51647 #malmen
    #sa2_y = 58.41102

    if coordinates_airport1 == None:
        x_ap1, y_ap1 = (15.670330652, 58.40499838) #coordinates linköping
    else:
        _ap1, y_ap1 = coordinates_airport1
    if coordinates_airport2 == None:
        x_ap2, y_ap2 = (15.51647, 58.41102) #coordinates_airport2
    else:
        x_ap2, y_ap2 = coordinates_airport2

    x_p, y_p = coordinates_airplane
    
    xmin, ymin, xmax, ymax= map.total_bounds
    # how many cells across and down
    n_cells = 30
    cell_size = (xmax-xmin)/n_cells
    
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
            

    #airports = gpd.GeoDataFrame(airport_cells, columns=['geometry'])
    #cell = gpd.GeoDataFrame(grid_cells, columns=['geometry'])

    # ?? needed ??
    #ax = cell.plot(facecolor="None", edgecolor='black')
    #airports.plot(ax=ax, facecolor="black", edgecolor='black')

    return grid_cells, airport_cells, plane_cells


# %%

fig, ax = plt.subplots(1, figsize=(10,10))

def animate(i):
    grid_cells, airport_cells, plane_cells = draw_map_with_locations(map, coordinates_airplane=(15.670330652, 58.40499838))

    ax2 = map.plot(markersize=.1, figsize=(12, 8), column='VDNAMN', cmap='jet', ax=ax)
    plt.autoscale(False)

    cell = gpd.GeoDataFrame(grid_cells, columns=['geometry'])
    cell.plot(ax=ax2, facecolor="none", edgecolor='grey')

    airports = gpd.GeoDataFrame(airport_cells, columns=['geometry'])
    airports.plot(ax=ax2, facecolor="black", edgecolor='black')

    plane = gpd.GeoDataFrame(plane_cells, columns=['geometry'])
    plane.plot(ax=ax2, facecolor="white", edgecolor='black')


#fig = plt.figure(figsize=(12, 8))
ani = animation.FuncAnimation(fig, animate, interval=1000) 
plt.show()


if __name__ == "__main__":
    monitor()
    plt.close()