import numpy as np
from stl import mesh
import datetime

from compute_inside import inside_z


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
