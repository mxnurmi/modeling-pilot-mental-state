import random
from math import dist
from typing import Tuple

EPSILON = 1e-3
# START_FUEL = 20  # also max fuel
#STRESS = 0
# MALMEN_LOCATION = (1, 1)  # (7, 9)
#LINKOPING_LOCATION = (5, 5)
#SIZE = (8, 8)
#WIND_PROB = 1

#  TODO: TO BE ADDED:
# - fuel dumb at 40% way where you only have enough fuel for return
# - Wind changes midway to favor original airport
# - Plane is ordered to return (DIFFICULT! -> do one of the above instead)
# - DO THIS:
# - Correct amount of fuel becomes uncertain (NOTE: Increasing uncertainty should be linked to stress!)

NUM_SIMS = 40000

def run_scenario(number="one"):
    if number == "one":
        """Simple condition (small domain and no uncertainty from changing conditions"""
        # easy scenario where plane has to land into other end of the map with plenty of fuel to spare
        init_scenario(wind=1, fuel_amount=11, fuel_keep_chance=1, n=7, airport1_coor=(2,2), airport2_coor=(4,5))
    elif number == "LARGE":
        """Larger domain condition"""
        init_scenario(wind=1, fuel_amount=22, fuel_keep_chance=1, n=9, airport1_coor=(2,3), airport2_coor=(7,7))
    elif number == "two":
        """Changing wind condition"""
        init_scenario(wind=0.85, fuel_amount=11, fuel_keep_chance=1, n=7, airport1_coor=(2,2), airport2_coor=(4,5))
    elif number == "three":
        init_scenario(wind=1, fuel_amount=11, fuel_keep_chance=0.95, n=7, airport1_coor=(2,2), airport2_coor=(4,5))
    elif number == "four":
        init_scenario(wind=0.85, fuel_amount=11, fuel_keep_chance=0.95, n=7, airport1_coor=(2,2), airport2_coor=(4,5))


def init_scenario(wind=None, fuel_amount=None, fuel_keep_chance=None, fuel_dumb_amount=None, n=None, airport1_coor=None, airport2_coor=None):
    # Maximum punishment
    global MAX_PUNISH
    MAX_PUNISH = -1000

    # Map size
    if n == None:
        n = random.randint(4, 9)
    global SIZE
    if type(n) == Tuple:
        SIZE = n
    else:
        SIZE = (n,n)

    # Start location coordinates
    global LINKOPING_LOCATION
    if airport1_coor == None:
        lin_x = random.randint(0, n-1)
        lin_y = random.randint(0, n-1)
        LINKOPING_LOCATION = (lin_x, lin_y)
    else:
        if type(airport1_coor) is not tuple:
            raise TypeError("Airport coordinates must be of type tuple")
        LINKOPING_LOCATION = airport1_coor

    # Landing location coordinates
    global MALMEN_LOCATION
    if airport2_coor == None:
        MALMEN_LOCATION = LINKOPING_LOCATION
        while dist(MALMEN_LOCATION, LINKOPING_LOCATION) <= 2:
            # Make sure we have enough distnce
            # TODO: chek that this works as intended
            MALMEN_LOCATION = (random.randint(0, n-1), random.randint(0, n-1))
    else:
        if type(airport2_coor) is not tuple:
            raise TypeError("Airport coordinates must be of type tuple")
        MALMEN_LOCATION = airport2_coor

    # Wind change probability
    global WIND_PROB
    if wind == None:
        WIND_PROB = random.uniform(0.5, 1)
    else:
        WIND_PROB = wind

    # Fuel start amount
    global START_FUEL
    if fuel_amount == None:
        START_FUEL = random.randint(10, 20)
    else:
        START_FUEL = fuel_amount

    # Fuel keep chance
    global FUEL_PROB
    if fuel_keep_chance == None:
        FUEL_PROB = random.uniform(0.95, 1)
    else:
        FUEL_PROB = fuel_keep_chance

    # Fuel drop amount
    global DUMB_AMOUNT
    if fuel_dumb_amount == None:
        DUMB_AMOUNT = random.randint(2, 5)
    else:
        DUMB_AMOUNT = fuel_dumb_amount  # if one then the same as no drop


# TODO: Hacky way to circumvent a the fact that xplane_master.py has difficulties initiating config FIX!:
print("CONFIG FILE HAS A SETUP RUNNING, UNCOMMENT IF NEEDED")
init_scenario(wind=1, fuel_amount=11, fuel_keep_chance=1, n=(21, 10), airport1_coor=(14,6), airport2_coor=(5,6))
