import numpy as np
from stl import mesh
from numba import njit, prange

from compute_intersections import intersections_z

#testing change

def sample_nonparallel(stl_mesh, points, inside_mesh_results):
	for ndx in range(points.shape[0]):
		intersections = intersections_z(points[ndx], stl_mesh.vectors)
		result = intersections % 2
		if result != 0:
			inside_mesh_results[ndx] = 1.0
		if intersections != 0:
			print(intersections, inside_mesh_results[ndx])

def sample_mesh(stl_mesh, num_samples):
	#create the sample coordinates
	exes	= stl_mesh.vectors[:,0]
	exe_min = np.amin(exes)
	exe_max = np.amax(exes)
	whys 	= stl_mesh.vectors[:,1]
	why_min = np.amin(whys)
	why_max = np.amax(whys)
	zees 	= stl_mesh.vectors[:,2]
	zee_min = np.amin(zees)
	zee_max = np.amax(zees)
	
	points = np.random.rand(1000, 3)
	points[:,0] = points[:,0] * (exe_max - exe_min)
	points[:,0] = points[:,0] + exe_min
	points[:,1] = points[:,1] * (why_max - why_min)
	points[:,1] = points[:,1] + why_min
	points[:,2] = points[:,2] * (zee_max - zee_min)
	points[:,2] = points[:,2] + zee_min
	
	inside_mesh_results = np.zeros(points.shape[0])
	sample_nonparallel(stl_mesh, points, inside_mesh_results)