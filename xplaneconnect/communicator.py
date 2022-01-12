import sys
import time
import xpc
from datetime import datetime

def monitor():
    global last_update

    with xpc.XPlaneConnect() as client:
        while True:
            

            last_update = datetime.now()
            posi = client.getPOSI();
            ctrl = client.getCTRL();

            print("Loc: (%4f, %4f, %4f) Aileron:%2f Elevator:%2f Rudder:%2f\n"\
               % (posi[0], posi[1], posi[2], ctrl[1], ctrl[0], ctrl[2]))
            
            dref = "sim/weather/wind_speed_kt[1]"
            values = 30
            #dref= "sim/cockpit/switches/gear_handle_status"
            client.sendDREF(dref, values)
            value = client.getDREF(dref)
            print(value)
            time.sleep(0.8)

if __name__ == "__main__":
    monitor()