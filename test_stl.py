import os
import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
#for performance testing
import datetime

figure = pyplot.figure()
axes = mplot3d.Axes3D(figure)

the_mesh = mesh.Mesh.from_file(os.path.join("data", "suzanne.stl"))

axes.add_collection3d(mplot3d.art3d.Poly3DCollection(the_mesh.vectors))

#have to scale the mesh
scale = the_mesh.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)

pyplot.show()

#from compute_intersections import intersections_z, intersections_z_alt, intersections_z_nonparallel
from sample_mesh import sample_mesh

sample_mesh(the_mesh, 100)

'''
# -- TEST CORRECTNESS -- 
exes	= the_mesh.vectors[:,0]
exe_min = np.amin(exes)
exe_max = np.amax(exes)
whys 	= the_mesh.vectors[:,1]
why_min = np.amin(whys)
why_max = np.amax(whys)
zees 	= the_mesh.vectors[:,2]
zee_min = np.amin(zees)
zee_max = np.amax(zees)
iter = 0
while iter < 1000:
	point = (np.random.uniform(exe_min, exe_max), np.random.uniform(why_min, why_max), np.random.uniform(zee_min, zee_max))

	intersections_nonparallel = intersections_z_nonparallel(point, the_mesh.vectors)
	intersections = intersections_z(point, the_mesh.vectors)
	intersections_alt = intersections_z_alt(point, the_mesh.vectors)
	if intersections_nonparallel != intersections or intersections != intersections_alt or intersections_alt != intersections_nonparallel:
		print(intersections_nonparallel, intersections, intersections_alt)
	iter += 1
print("done checking correctness")
'''
'''
# -- TESTING PERFORMANCE OF INTERSECTION CALCULATIONS --
#the cuda versions are way faster
# i think the first implementation may be faster than the alt
num_iter = 10000
exes	= the_mesh.vectors[:,0]
exe_min = np.amin(exes)
exe_max = np.amax(exes)
whys 	= the_mesh.vectors[:,1]
why_min = np.amin(whys)
why_max = np.amax(whys)
zees 	= the_mesh.vectors[:,2]
zee_min = np.amin(zees)
zee_max = np.amax(zees)

start_time = datetime.datetime.now()
iter = 0
while iter < num_iter:
	point = (np.random.uniform(exe_min, exe_max), np.random.uniform(why_min, why_max), np.random.uniform(zee_min, zee_max))
	intersections = intersections_z_nonparallel(point, the_mesh.vectors)
	iter += 1
end_time = datetime.datetime.now()
print("done testing nonparallel time")
print(end_time - start_time)

start_time = datetime.datetime.now()
iter = 0
while iter < num_iter:
	point = (np.random.uniform(exe_min, exe_max), np.random.uniform(why_min, why_max), np.random.uniform(zee_min, zee_max))
	intersections = intersections_z(point, the_mesh.vectors)
	iter += 1
end_time = datetime.datetime.now()
print("done testing GPU parallel time")
print(end_time - start_time)

start_time = datetime.datetime.now()
iter = 0
while iter < num_iter:
	point = (np.random.uniform(exe_min, exe_max), np.random.uniform(why_min, why_max), np.random.uniform(zee_min, zee_max))
	intersections = intersections_z_alt(point, the_mesh.vectors)
	iter += 1
end_time = datetime.datetime.now()
print("done testing GPU parallel time")
print(end_time - start_time)
'''

