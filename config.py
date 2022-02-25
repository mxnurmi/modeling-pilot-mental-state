import random
from math import dist

EPSILON = 1e-3
#START_FUEL = 20  # also max fuel
#STRESS = 0
#MALMEN_LOCATION = (1, 1)  # (7, 9)
#LINKOPING_LOCATION = (5, 5)
#SIZE = (8, 8)
#WIND_PROB = 1


# MOVE TO CONFIG
def init_scenario(wind=None):
    n = random.randint(4, 9)
    global SIZE
    SIZE = (n, n)

    lin_x = random.randint(0, n-1)
    lin_y = random.randint(0, n-1)
    global LINKOPING_LOCATION
    LINKOPING_LOCATION = (lin_x, lin_y)

    global MALMEN_LOCATION
    MALMEN_LOCATION = LINKOPING_LOCATION
    print(dist(MALMEN_LOCATION, LINKOPING_LOCATION))
    print(MALMEN_LOCATION)
    print(LINKOPING_LOCATION)
    while dist(MALMEN_LOCATION, LINKOPING_LOCATION) <= 2:
        # TODO: chek that this works as intended
        MALMEN_LOCATION = (random.randint(0, n-1), random.randint(0, n-1))
    
    global WIND_PROB
    if wind == None:
        WIND_PROB = random.uniform(0.5, 1)
    else:
        WIND_PROB = wind

    global START_FUEL
    START_FUEL = random.randint(8, 20)

