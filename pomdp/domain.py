import pomdp_py

# https://h2r.github.io/pomdp-py/html/_modules/pomdp_problems/tag/domain/action.html#TagAction
class PlaneAction(pomdp_py.Action):
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, PlaneAction):
            return self.name == other.name
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return "PlaneAction(%s)" % self.name


class MoveAction(PlaneAction):
    EAST = (1, 0)  # x is horizontal; x+ is right. y is vertical; y+ is up.
    WEST = (-1, 0)
    NORTH = (0, 1)  # are these correct? -> changed from original
    SOUTH = (0, -1)  # are these correct? -> changed from original

    def __init__(self, motion, name):
        if motion not in {MoveAction.EAST, MoveAction.WEST,
                          MoveAction.NORTH, MoveAction.SOUTH}:
            raise ValueError("Invalid move motion %s" % motion)
        self.motion = motion
        super().__init__("move-%s" % str(name))


MoveEast = MoveAction(MoveAction.EAST, "EAST")
MoveWest = MoveAction(MoveAction.WEST, "WEST")
MoveNorth = MoveAction(MoveAction.NORTH, "NORTH")
MoveSouth = MoveAction(MoveAction.SOUTH, "SOUTH")

# waiting also checks for wind


class WaitAction(PlaneAction):
    def __init__(self):
        super().__init__("check-wind")


class LandAction(PlaneAction):
    def __init__(self):
        super().__init__("land")


# https://h2r.github.io/pomdp-py/html/_modules/pomdp_problems/tag/domain/state.html#TagState
class PlaneState(pomdp_py.State):
    def __init__(self, coordinates, position, wind, fuel):
        self.coordinates = coordinates
        self.position = position
        self.wind = wind
        self.fuel = fuel

    def __hash__(self):
        return hash((self.coordinates, self.position, self.wind, self.fuel)) #self.location

    def __eq__(self, other):
        if isinstance(other, PlaneState):
            # checks if the other state has identical name to this
            return self.coordinates == other.coordinates\
                and self.position == other.position\
                and self.wind == other.wind\
                and self.fuel == other.fuel
        return False

    def __str__(self):
        return 'State(%s| %s, %s, %s)' % (str(self.coordinates), #TODO: Is | correct in string format?
                                      str(self.position),
                                      str(self.wind),
                                      str(self.fuel)
                                      )

    def __repr__(self):
        return str(self)


class PlaneObservation(pomdp_py.Observation):
    def __init__(self, wind):
        self.wind = wind

    def __hash__(self):
        return hash(self.wind)

    def __eq__(self, other):
        if isinstance(other, PlaneObservation):
            return self.wind == other.wind
        return False

    def __str__(self):
        return str(self.wind)

    def __repr__(self):
        return str(self)
