import numpy as np
from stl import mesh
import datetime

from compute_inside import inside_z


def sample_mesh(stl_mesh, num_samples):
	print("sampling mesh with %i samples" % num_samples)
	#create the sample coordinates
	exes	= stl_mesh.vectors[:,:,0]
	exe_min = np.amin(exes)
	exe_max = np.amax(exes)
	whys 	= stl_mesh.vectors[:,:,1]
	why_min = np.amin(whys)
	why_max = np.amax(whys)
	zees 	= stl_mesh.vectors[:,:,2]
	zee_min = np.amin(zees)
	zee_max = np.amax(zees)
	
	print(exe_min, exe_max, why_min, why_max, zee_min, zee_max)
	
	points = np.random.rand(num_samples, 3)
	points[:,0] = points[:,0] * (exe_max - exe_min)
	points[:,0] = points[:,0] + exe_min
	points[:,1] = points[:,1] * (why_max - why_min)
	points[:,1] = points[:,1] + why_min
	points[:,2] = points[:,2] * (zee_max - zee_min)
	points[:,2] = points[:,2] + zee_min
	
	inside_results = np.zeros(points.shape[0])
	faces = np.ascontiguousarray(stl_mesh.vectors)
	
	start_time = datetime.datetime.now()
	inside_z(points, faces, inside_results)
	end_time = datetime.datetime.now()
	print(end_time - start_time)
	
	return points, inside_results

#this works by taking the face centers, scaling them up and down, and using all those as the samples
def sample_mesh_selective(stl_mesh):
	points = np.zeros((stl_mesh.vectors.shape[0], stl_mesh.vectors.shape[2]), dtype = np.float32)
	for ndx in range(points.shape[0]):
		avg = stl_mesh.vectors[ndx][0] + stl_mesh.vectors[ndx][1] + stl_mesh.vectors[ndx][2]
		avg = avg * (1/3)
		points[ndx] = avg
	scaled_down = points * 0.9
	scale_up = points * 1.1
	print(points.shape, points.dtype)
	print(scaled_down.shape, scaled_down.dtype)
	print(scale_up.shape, scale_up.dtype)
	points = np.concatenate((points, scaled_down, scale_up))
	'''
	#print("sampling mesh with %i samples" % num_samples)
	#create the sample coordinates
	exes	= stl_mesh.vectors[:,:,0]
	exe_min = np.amin(exes)
	exe_max = np.amax(exes)
	whys 	= stl_mesh.vectors[:,:,1]
	why_min = np.amin(whys)
	why_max = np.amax(whys)
	zees 	= stl_mesh.vectors[:,:,2]
	zee_min = np.amin(zees)
	zee_max = np.amax(zees)
	
	print(exe_min, exe_max, why_min, why_max, zee_min, zee_max)
	
	num_samples = points.shape[0]
	
	points = np.random.rand(num_samples, 3)
	points[:,0] = points[:,0] * (exe_max - exe_min)
	points[:,0] = points[:,0] + exe_min
	points[:,1] = points[:,1] * (why_max - why_min)
	points[:,1] = points[:,1] + why_min
	points[:,2] = points[:,2] * (zee_max - zee_min)
	points[:,2] = points[:,2] + zee_min
	'''
	inside_results = np.zeros(points.shape[0])
	faces = np.ascontiguousarray(stl_mesh.vectors)
	
	start_time = datetime.datetime.now()
	inside_z(points, faces, inside_results)
	end_time = datetime.datetime.now()
	print(end_time - start_time)
	
	return points, inside_results