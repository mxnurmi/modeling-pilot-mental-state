# %%
import matplotlib.pyplot as plt


def visualize_plane_problem(width, height, plane_location, airport_location1, airport_location2):

    # TODO: create this in two loops, first one adds numbers to lists and second one copies them. 
    # Then when we have plane or airport location we just count when we hit that and add zero

    coordinates = []
    for y in range(0, height):
        row = []
        for x in range(0, width):
            if (x,y) == plane_location:
                c = 0
            elif ((x,y) == airport_location1) or ((x,y) == airport_location2):
                c = 1
            else:
                c = 2
            row.append(c)
        coordinates.append(row)

    plt.imshow(coordinates,  origin='lower', cmap='gray')

    # TODO: change tick labels to x * y 
    # # where u can be found from plt.yticks(np.arange(y.min(), y.max(), 1))

    plt.show()

# %%

visualize_plane_problem(31, 16, plane_location=(7, 5), airport_location1=(1,1), airport_location2=(7, 9))

# %%
