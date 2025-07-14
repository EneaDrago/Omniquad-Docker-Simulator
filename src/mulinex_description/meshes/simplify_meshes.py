#!/bin/python

import pymeshlab
import os
import glob

ms = pymeshlab.MeshSet()
# load all the meshes files in the package mulinex_description 
# and apply the filter 'convex_hull' to each of them
for filename in glob.glob("*.stl"):
    ms.load_new_mesh(filename)
    # skip if it starts with simpler_
    if filename.startswith('simpler_'):
        continue
    ms.apply_filter('generate_convex_hull')
    ms.save_current_mesh('simpler_'+filename)
    print(filename + ' simplified and saved as simpler_'+ filename)
print('*'*20)
print('All done! Don\'t forget to edit the links.xacro file to use the new meshes')

