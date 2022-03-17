import random
from math import dist

EPSILON = 1e-3
#START_FUEL = 20  # also max fuel
#STRESS = 0
#MALMEN_LOCATION = (1, 1)  # (7, 9)
#LINKOPING_LOCATION = (5, 5)
#SIZE = (8, 8)
#WIND_PROB = 1

#  TODO: TO BE ADDED:
# - fuel dumb at 40% way where you only have enough fuel for return
# - Wind changes midway to favor original airport
# - Plane is ordered to return (DIFFICULT! -> do one of the above instead)
# - DO THIS: 
# - Correct amount of fuel becomes uncertain (NOTE: Increasing uncertainty should be linked to stress!)

def init_scenario(wind=None, fuel=None, dumb_amount=None, n=None):
    global MAX_PUNISH
    MAX_PUNISH = -10000
    if n == None:
        n = random.randint(4, 9)
    global SIZE
    SIZE = (n, n)

    lin_x = random.randint(0, n-1)
    lin_y = random.randint(0, n-1)
    global LINKOPING_LOCATION
    LINKOPING_LOCATION = (lin_x, lin_y)

    global MALMEN_LOCATION
    MALMEN_LOCATION = LINKOPING_LOCATION

    while dist(MALMEN_LOCATION, LINKOPING_LOCATION) <= 2:
        # TODO: chek that this works as intended
        MALMEN_LOCATION = (random.randint(0, n-1), random.randint(0, n-1))
    
    global WIND_PROB
    if wind == None:
        WIND_PROB = random.uniform(0.5, 1)
    else:
        WIND_PROB = wind

    global START_FUEL
    START_FUEL = random.randint(10, 20)

    global FUEL_PROB
    if fuel == None:
        FUEL_PROB = random.uniform(0.95, 1)
    else:
        FUEL_PROB = fuel

    global DUMB_AMOUNT
    if dumb_amount == None:
        DUMB_AMOUNT = random.randint(1,5)
    else:
        DUMB_AMOUNT = dumb_amount
    
