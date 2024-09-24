# -*- coding: utf-8 -*-
"""
Code example for extending glacier cross sections until they reach the glacier outline.
This was used in the LIA glacier surface reconstruction by Reinthaler & Paul 2024 (https://doi.org/10.1016/j.geomorph.2024.109321)

@author: Adrien Wehrlé & Johannes Reinthaler

For best usage, use short cross section to start, we recommend around 10 m. 
"""

import shapely

from shapely.geometry import LineString
import geopandas as gpd
import os
import matplotlib.pyplot as plt
import numpy as np

#function for extrapolating line
def getExtrapoledLine(p1,p2):
    'Creates a line extrapoled in p1->p2 direction'
    EXTRAPOL_RATIO = 1e10
    a = p1
    b = (p1[0]+EXTRAPOL_RATIO*(p2[0]-p1[0]), p1[1]+EXTRAPOL_RATIO*(p2[1]-p1[1]) )
    
    a2 = p2
    b2 = (p2[0]+EXTRAPOL_RATIO*(p1[0]-p2[0]), p2[1]+EXTRAPOL_RATIO*(p1[1]-p2[1]) )
    
    return LineString([b,b2])


os.chdir("file dictionary")
shape_path = "glacier outlines shapefile file path"
cross_path = "glacier cross section shapefile file path"
poly_file = gpd.read_file(shape_path)
crosssection_file = gpd.read_file(cross_path)

extended_lines = []

for glacier_id in poly_file.id1:        #change id1 for the ID column name

    glacier_poly = poly_file[poly_file.id1 == glacier_id].iloc[0].geometry.buffer(0)
    glacier_css = crosssection_file[crosssection_file.id1 == glacier_id].geometry
  
    for cs in glacier_css:
        
        l_coords = list(cs.coords)
        long_line = getExtrapoledLine(*l_coords) #we use the last two points; for first two [2:]
        
        cs_centroid = cs.interpolate(cs.length/2)
        
        if glacier_poly.intersects(long_line):
            intersection_points = glacier_poly.intersection(long_line)
            
            if type(intersection_points) == shapely.geometry.multilinestring.MultiLineString:
                
                dists = []
                
                for i, line in enumerate(intersection_points):
                    
                    line_centroid = line.interpolate(line.length/2)
                    dist = np.sqrt((line_centroid.coords[0][0] - cs_centroid.coords[0][0]) ** 2
                                   + (line_centroid.coords[0][1] - cs_centroid.coords[0][1]) **2)
                    dists.append(dist)
                
                # closest centroid to line centroid
                correct_id = np.where(np.array(dists) == np.nanmin(dists))[0][0]
                l_coords = list(intersection_points[correct_id].coords) 
                
            else:
                
                l_coords = list(intersection_points.coords)

        extended_lines.append(LineString(l_coords))
        
  # %%
#Plot the results for a specific glacier
from matplotlib.lines import Line2D
from shapely.geometry import LineString, Point

# Specify the glacier ID
glacier_id = 10  # Change this value to any desired ID

# Filter the glacier polygon and cross-sections for the specified glacier_id
glacier_poly = poly_file[poly_file.id1 == glacier_id].iloc[0].geometry.buffer(0)
glacier_css = crosssection_file[crosssection_file.id1 == glacier_id].geometry

# Plot the glacier polygon outline
x, y = glacier_poly.exterior.xy
plt.plot(x, y, color="blue", label=f"Glacier Polygon (id1={glacier_id})")

# Find extended lines with centers within the glacier polygon
best_extended_lines = []

for line in extended_lines:
    # Calculate the center point of the extended line
    center_point = line.interpolate(line.length / 2)
    
    # Check if the center point is within the glacier polygon
    if glacier_poly.contains(center_point):
        best_extended_lines.append(line)

# Plot the extended cross-sections with centers within the glacier polygon
for extended_line in best_extended_lines:
    ext_x, ext_y = extended_line.xy
    plt.plot(ext_x, ext_y, color="orange", linewidth=1, label="Extended Cross-section")

# Plot the original cross-sections
for cs in glacier_css:
    cs_x, cs_y = cs.xy
    plt.plot(cs_x, cs_y, color="green", linewidth=2, label="Original Cross-section")

# Create custom legend handles
original_cross_section_handle = Line2D([0], [0], color="green", linewidth=2)
extended_cross_section_handle = Line2D([0], [0], color="orange", linewidth=1)
glacier_polygon_handle = Line2D([0], [0], color="blue", linewidth=2)

# Add labels and show the plot with custom legend
plt.legend(handles=[glacier_polygon_handle, original_cross_section_handle, extended_cross_section_handle],
           labels=[f"Glacier Polygon (id1={glacier_id})", "Cross-sections", "Extended Cross-sections"])
plt.xlabel("Easting")
plt.ylabel("Northing")
plt.title(f"Glacier Polygon and Cross-sections (id1={glacier_id})")
plt.show()

      
# %%
#Export the extended cross secions
newdata = []
results1 = gpd.GeoDataFrame(newdata, geometry = extended_lines, crs="EPSG:32640")
results1.to_file("Output.shp", driver="ESRI Shapefile",encoding="utf-8")

