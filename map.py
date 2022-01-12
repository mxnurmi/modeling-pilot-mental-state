# %%

import geopandas as gpd
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import numpy as np
import shapely

#map = gpd.read_file("./Sweden_shapefile/se_1km.shp")

# Change projection
import pyproj
from functools import partial
from shapely.ops import transform

# %%

map2 = gpd.read_file("./alla_valdistrikt/alla_valdistrikt.shp")

fig, ax = plt.subplots(figsize=(50, 50))

print(map2)
map2.plot(ax=ax)
# %%
map2.head()

map2.explore()
# %%
#tes1 = map2[(map2.VDNAMN == "Kärna")]

# Only pick regions near the airport
locations = ["Kärna", "S:t Lars", "Domkyrko \d* \(",
             "Berga \d \(", "Slaka", "Ryd \d", "Skäggetorp", "Landeryd 1",
             "Johannelund \d", "Landeryd \d"]

tes1 = map2[map2['VDNAMN'].str.contains('|'.join(locations))]

df = DataFrame(map2.VDNAMN)

tes1.explore()
# %%
print(tes1.head())
# %%
fig, ax = plt.subplots(figsize=(30, 40))
#tes1.plot(ax=ax)

# %%

#tes1.explore()

# %%
sa_x = 15.670330652 #15.67
sa_y = 58.40499838

sa2_x = 15.51647 #malmen
sa2_y = 58.41102

tes1 = tes1.to_crs({'init': 'epsg:4326'})
tes1.plot()

# %%
xmin, ymin, xmax, ymax= tes1.total_bounds
# how many cells across and down
n_cells=30
cell_size = (xmax-xmin)/n_cells
# projection of the grid
#crs = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
# create the cells in a loop
grid_cells = []
airport_cells = []
# TODO: We can also use this to create a separate matrix for visualizations
for x0 in np.arange(xmin, xmax+cell_size, cell_size):
    for y0 in np.arange(ymin, ymax+cell_size, cell_size):
        # bounds
        x1 = x0-cell_size
        y1 = y0+cell_size

        #x1 = left part
        #y1 = top part

        grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))

        if (sa_x > x1) and (sa_x < x0) and (sa_y > y0) and (sa_y < y1):
            airport_cells.append(shapely.geometry.box(x0, y0, x1, y1))

        if (sa2_x > x1) and (sa2_x < x0) and (sa2_y > y0) and (sa2_y < y1):
            airport_cells.append(shapely.geometry.box(x0, y0, x1, y1))


ax = tes1.plot(markersize=.1, figsize=(12, 8), column='VDNAMN', cmap='jet')
plt.autoscale(False)

cell = gpd.GeoDataFrame(grid_cells, columns=['geometry'])
cell.plot(ax=ax, facecolor="none", edgecolor='grey')

airports = gpd.GeoDataFrame(airport_cells, columns=['geometry'])
airports.plot(ax=ax, facecolor="black", edgecolor='black')

#ax.axis("off")

# %%

print(len(grid_cells)/31) # Height of grid

# Lat, lon: 58.40499838 15.670330652
# UTM: 539173.94569427 6473996.5191658

# %%

# JUST THE GRID
airports = gpd.GeoDataFrame(airport_cells, columns=['geometry'])
cell = gpd.GeoDataFrame(grid_cells, columns=['geometry'])

ax = cell.plot(facecolor="None", edgecolor='black')
airports.plot(ax=ax, facecolor="black", edgecolor='black')


# %%
# korkeus: 15 leveys: 31
# malmö: 9,7
# linkö: 8, 21