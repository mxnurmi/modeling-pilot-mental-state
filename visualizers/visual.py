# %%
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def create_colored_coordinates_for_plane(width, height, plane_location, airport_location1, airport_location2):
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
    
    return coordinates

def get_coordinates(width, height, plane_location, airport_location1, airport_location2):
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

    return coordinates

def visualize_plane_problem(width, height, plane_location, airport_location1, airport_location2, plot=True):

    # TODO: create this in two loops, first one adds numbers to lists and second one copies them. 
    # Then when we have plane or airport location we just count when we hit that and add zero
    
    if plot == False:
        return None

    fig = plt.figure( figsize=(8,8) )
    coordinates = get_coordinates(width, height, plane_location, airport_location1, airport_location2)
    im = plt.imshow(coordinates, origin='lower', cmap='gray')

    def animate_func(i):
        #print(i)
        #if i == 5:
        #    print(f'{i} == {FRAMES_NUM}; closing!')
            #plt.close(fig)

        coordinates = get_coordinates(width, height, plane_location, airport_location1, airport_location2)
        im.set_array(coordinates)
        #return [im]

    # TODO: crashed and landed state should be somehow handled in a better way. Now the plane just wanishes and airports change color :/

    #ax.text(3, 8, 'boxed italics text in data coords', style='italic',
    #bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

    #plot_loc_x = -1
    #plot_loc_y = -(2/3 * plot_size[1])
    
    #plt.imshow(coordinates,  origin='lower', cmap='gray')
    #plt.text(plot_loc_x, plot_loc_y, text, fontsize = 22, 
    #     bbox = dict(facecolor = 'grey', alpha = 0.5))

    # use this: https://pretagteam.com/question/how-to-add-legend-to-imshow-in-matplotlib
    #plt.pause(0.001) # TODO: This is poor way to animate as it captures focus

    anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = 11,
                               interval = 100, # in ms
                               )
    plt.show()

# %%
if __name__ == "__main__":
    #visualize_plane_problem(31, 15, plane_location=(21,8), airport_location1=(7,9), airport_location2=(21,8), plot=True)

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    grid_size = [10, 10]
    grid = np.random.randint(low=0, high=256, size=grid_size, dtype=np.uint8)
    im = ax.imshow(grid, cmap='gray', vmin=0, vmax=255)

    # Animation settings
    def animate(frame):
        if frame == FRAMES_NUM:
            print(f'{frame} == {FRAMES_NUM}; closing!')
            plt.close(fig)
        else:
            print(f'frame: {frame}') # Debug: May be useful to stop
            grid = np.random.randint(low=0, high=256, size=grid_size, dtype=np.uint8)
            im.set_array(grid)

    INTERVAL = 100
    FRAMES_NUM = 10

    anim = animation.FuncAnimation(fig, animate, interval=INTERVAL, 
                                frames=FRAMES_NUM+1, repeat=False)

    plt.show()

# %%
