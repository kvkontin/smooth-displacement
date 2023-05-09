# Delaunay Triangulation based smooth displacement
# Author: Kaapo Kontinen

# NOTE: To use third party packages with Blender, delete the Python folder in Blender's directory.
# Blender will fall back on system Python installation as long as it matches major version.
# More information: https://docs.blender.org/api
#   /blender_python_api_2_69_release/info_tips_and_tricks.html#bundled-python-extensions

# How to use this file:
# 1. Import the object to be deformed as a Mesh Object.
# 2. Create a new Collection for the anchors.
# 3. In the Collection, create any number of any type of Object. These are the anchors.
# 4. For each anchor, create any type of Object. Make the anchor its parent. These are the handles.
# 5. Edit the variables below to match your objects.
# 6. Run this script.


DEFORMED_OBJECT_NAME    = 'unspecified object name'
ANCHORS_NAME            = 'unspecified collection name'

import bpy
from mathutils import Vector
import numpy as np
from scipy.spatial import Delaunay

def smooth_function(fraction):
    if(fraction <= 0):
        return 0
    if(fraction >= 1):
        return 1
    return 3*fraction*fraction - 2*fraction*fraction*fraction

QHULL_OPTIONS = 'Qc Q12 QJ' # Required for Delaunay to accept coplanar points
DELAUNAY_MINIMUM_POINTS = 5 # Delaunay is inapplicable for sets smaller than 5 points

cloud = bpy.data.objects[DEFORMED_OBJECT_NAME]
points = np.array(cloud.data.vertices) # Object-space coordinates
matrix_cloud_to_world = cloud.matrix_world.inverted()

# Anchors are parentless objects in the ANCHORS_NAME collection
anchors = list(filter(lambda o: o.parent == None, bpy.data.collections[ANCHORS_NAME].objects))
delaunay_points = np.array([a.location for a in anchors])

# Compute which anchors are neighbors to each other
neighbors = [0]*len(anchors)
if(len(anchors) >= DELAUNAY_MINIMUM_POINTS):
    delaunay = Delaunay(delaunay_points, qhull_options=QHULL_OPTIONS)
    pointers, indices = delaunay.vertex_neighbor_vertices
    # The indices of neighbors of anchor k are indices[pointers[k]:pointers[k+1]].
    for k in range(len(anchors)):
        neighbors[k] = indices[pointers[k]:pointers[k+1]]
else:
    for k in range(len(anchors)):
        all_except_this = list(range(len(anchors)))
        all_except_this.pop(k)
        neighbors[k] = all_except_this

# For each point, compute displacement coefficients, then use them
for point in points:
    
    point_in_world = cloud.matrix_world @ point.co
    influence_coeffs = [1]*len(anchors)
    
    # find which enlarged cells contain this point
    for first_anchor_index in range(len(anchors)):
        first_coords = anchors[first_anchor_index].location
        in_cell = True
        for second_anchor_index in range(len(anchors)):
            if(first_anchor_index == second_anchor_index):
                continue
            second_coords = anchors[second_anchor_index].location
            delta_normalized = (second_coords-first_coords).normalized()
            outer_threshold = second_coords.dot(delta_normalized)

            dot = point_in_world.dot(delta_normalized)
            if(dot > outer_threshold):
                in_cell = False
                influence_coeffs[first_anchor_index] = 0
                break
            # end for second_anchor_index
            
        if(in_cell):
            
            # For each boundary of the cell, calculate an influence factor
            for neighbor_index in neighbors[first_anchor_index]:
                
                second_coords = anchors[neighbor_index].location
                delta_normalized = (second_coords-first_coords).normalized()
                inner_threshold = first_coords.dot(delta_normalized)
                outer_threshold = second_coords.dot(delta_normalized)
                
                fraction = (point_in_world.dot(delta_normalized)-inner_threshold) / (outer_threshold-inner_threshold)
                influence_curve = 1-smooth_function(fraction)
                influence_coeffs[first_anchor_index] *= influence_curve
            # end for boundary
        # end if(in_cell)
    # end for first_anchor_index
    
    # Normalize displacement coefficients
    coeffs_sum = sum(influence_coeffs)
    if(coeffs_sum != 0):
        influence_coeffs = [i/coeffs_sum for i in influence_coeffs]

    # Apply displacement
    new_in_world = Vector((0,0,0))
    for anchor_index in range(len(anchors)):
        anchor = anchors[anchor_index]
        end = anchor.children[0]

        point_in_anchor = anchor.matrix_world.inverted() @ cloud.matrix_world @ point.co
        displaced_in_world = end.matrix_world @ point_in_anchor
        new_in_world += influence_coeffs[anchor_index]*displaced_in_world
    # end for anchor_index
    
    point.co = matrix_cloud_to_world @ new_in_world
    
# end for point