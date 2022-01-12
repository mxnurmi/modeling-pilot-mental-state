import sys
from xplaneconnect import xpc
import simple_pid as PID

from datetime import datetime, timedelta
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import time

def normalize(value, min=-1, max=1):
    # if value = 700, and max = 20, return 20
    # if value = -200, and min = -20, return -20
    if (value > max):
        return max
    elif (value < min):
        return min
    else:
        return value
update_interval = 0.050 # seconds, 0.05 = 20 Hz
start = datetime.now()
last_update = start
# defining the initial PID values
P = 0.1 # PID library default = 0.2
I = P/10 # default = 0
D = 0 # default = 0
# initializing PID controllers
roll_PID = PID.PID(P, I, D)
pitch_PID = PID.PID(P, I, D)
altitude_PID = PID.PID(P, I, D)
# setting the desired values
# roll = 0 means wings level
# pitch = 2 means slightly nose up, which is required for level flight
desired_roll = 0
desired_pitch = 2
desired_altitude = 8000
# setting the PID set points with our desired values
roll_PID.SetPoint = desired_roll
pitch_PID.SetPoint = desired_pitch
altitude_PID.SetPoint = desired_altitude
x_axis_counters = [] #0, 1, 2, 3, etc. just basic x-axis values used for plotting
roll_history = []
pitch_history = []
altitude_history = []
roll_setpoint_history = []
pitch_setpoint_history = []
altitude_setpoint_history = []
plot_array_max_length = 300 # how many data points to hold in our arrays and graph
i = 1 # initialize x_axis_counter
# first the base app needs to be instantiated
app = pg.mkQApp("python xplane autopilot monitor")
# now the window itself is defined and sized
win = pg.GraphicsLayoutWidget(show=True)
win.resize(1000,600) #pixels
win.setWindowTitle("XPlane autopilot system control")
# we have 3 subplots
p1 = win.addPlot(title="roll",row=0,col=0)
p2 = win.addPlot(title="pitch",row=1,col=0)
p3 = win.addPlot(title="altitude", row=2, col=0)
# show the y grid lines to make it easier to interpret the graphs
p1.showGrid(y=True)
p2.showGrid(y=True)
p3.showGrid(y=True)
DREFs = ["sim/cockpit2/gauges/indicators/airspeed_kts_pilot",
        "sim/cockpit2/gauges/indicators/heading_electric_deg_mag_pilot",
        "sim/flightmodel/failures/onground_any",
        "sim/flightmodel/misc/h_ind"]

time.sleep(10)

def monitor():
    global i
    global last_update
    with xpc.XPlaneConnect() as client:
        while True:
            if (datetime.now() > last_update + timedelta(milliseconds = update_interval * 1000)):
                last_update = datetime.now()
                print(f"loop start - {datetime.now()}")
                posi = client.getPOSI();
                ctrl = client.getCTRL();
                multi_DREFs = client.getDREFs(DREFs)
                current_roll = posi[4]
                current_pitch = posi[3]
                current_hdg = multi_DREFs[1][0]
                current_altitude = multi_DREFs[3][0]
                current_asi = multi_DREFs[0][0]
                onground = multi_DREFs[2][0]
                # update the display
                pg.QtGui.QApplication.processEvents()
                # update outer loops first
                altitude_PID.update(current_altitude)
                # if alt=12000, setpoint = 10000, the error is 2000. if P=0.1, output will be 2000*0.1=200
                pitch_PID.SetPoint = normalize(altitude_PID.output, min=-15, max=10)
                # update PIDs
                roll_PID.update(current_roll)
                pitch_PID.update(current_pitch)
                # update control outputs
                new_ail_ctrl = normalize(roll_PID.output)
                new_ele_ctrl = normalize(pitch_PID.output)
                # if we reach our data limit set point, evict old data and add new.
                # this helps keep the graph clean and prevents it from growing infinitely
                if(len(x_axis_counters) > plot_array_max_length):
                    x_axis_counters.pop(0)
                    roll_history.pop(0)
                    roll_setpoint_history.pop(0)
                    pitch_history.pop(0)
                    pitch_setpoint_history.pop(0)
                    altitude_history.pop(0)
                    altitude_setpoint_history.pop(0)
                    x_axis_counters.append(i)
                    roll_history.append(current_roll)
                    roll_setpoint_history.append(desired_roll)
                    pitch_history.append(current_pitch)
                    pitch_setpoint_history.append(pitch_PID.SetPoint)
                    altitude_history.append(0)
                    altitude_setpoint_history.append(desired_altitude)
                # else, just add new. we are not yet at limit.
                else:
                    x_axis_counters.append(i)
                    roll_history.append(current_roll)
                    roll_setpoint_history.append(desired_roll)
                    pitch_history.append(current_pitch)
                    pitch_setpoint_history.append(pitch_PID.SetPoint)
                    altitude_history.append(0)
                    altitude_setpoint_history.append(desired_altitude)
                i = i + 1
                p1.plot(x_axis_counters, roll_history, pen=0, clear=True)
                p1.plot(x_axis_counters, roll_setpoint_history, pen=1)
                p2.plot(x_axis_counters, pitch_history, pen=0,clear=True)
                p2.plot(x_axis_counters, pitch_setpoint_history, pen=1)
                p3.plot(x_axis_counters, altitude_history, pen=0,clear=True)
                p3.plot(x_axis_counters, altitude_setpoint_history, pen=1)
                # sending actual control values to XPlane
                ctrl = [new_ele_ctrl, new_ail_ctrl, 0.0, -998] # ele, ail, rud, thr. -998 means don't change
                client.sendCTRL(ctrl)
                output = f"current values --    roll: {current_roll: 0.3f},  pitch: {current_pitch: 0.3f}"
                output = output + "\n" + f"PID outputs    --    roll: {roll_PID.output: 0.3f},  pitch: {pitch_PID.output: 0.3f}"
                output = output + "\n"
                print(output)
if __name__ == "__main__":
    monitor()