# %%
import matplotlib.pyplot as plt

def visualize_plane_problem(width, height, plane_location, airport_location1, airport_location2, plot):

    # TODO: create this in two loops, first one adds numbers to lists and second one copies them. 
    # Then when we have plane or airport location we just count when we hit that and add zero
    
    if plot == False:
        return None

    # TODO: crashed and landed state should be somehow handled in a better way. Now the plane just wanishes and airports change color :/
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

    #ax.text(3, 8, 'boxed italics text in data coords', style='italic',
    #bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

    #plot_loc_x = -1
    #plot_loc_y = -(2/3 * plot_size[1])
    
    plt.imshow(coordinates,  origin='lower', cmap='gray')
    #plt.text(plot_loc_x, plot_loc_y, text, fontsize = 22, 
    #     bbox = dict(facecolor = 'grey', alpha = 0.5))


    # use this: https://pretagteam.com/question/how-to-add-legend-to-imshow-in-matplotlib
    plt.show()

# %%
if __name__ == "__main__":
    visualize_plane_problem(31, 15, plane_location=(21,8), airport_location1=(7,9), airport_location2=(21,8))

# %%
