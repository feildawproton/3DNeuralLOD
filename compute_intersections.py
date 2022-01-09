import numpy
from stl import mesh
from numba import cuda, float32
import numpy as np
import math 

#The fastest implementation, on my computer at least, seems to be intersections_z()

THREADS_PER_BLOCK = 64

def volume_cpu(a, b, c, d):
	a_d = a - d
	b_d = b - d
	c_d = c - d
	crss = np.cross(b_d, c_d)
	numerator = np.dot(a_d, crss)
	return numerator / 6

#cast the ray in the z direction
#this is the nonparallel version
#per my testing the parallel version is significantly faster
def intersections_z_nonparallel(point, faces):
	results = np.zeros(faces.shape[0])  #should this be a different dtype?  by default its a float
	z_limt = np.amin(faces[:,2])
	for ndx, face in enumerate(faces):
		
		# -- STEP 1 -- 
		#determine if the line segment crosses the plane containing the triangle
		#got this from stack overflow: https://stackoverflow.com/questions/53962225/how-to-know-if-a-line-segment-intersects-a-triangle-in-3d-space
		a = face[0]
		b = face[1]
		c = face[2]
		d = point
		e = (point[0], point[1], z_limt)
		Td = volume_cpu(a, b, c, d)
		Te = volume_cpu(a, b, c, e)
		
		#if Td and Te have the same sign then the points lie on the same side of the plane
		#if they have opposite sign then they cross the plane
		#if one of them is zero that point lies on the plane
		#so this should be simply:
		if Td*Te <= 0.0:
			# -- STEP 2 --
			#solve the point in triangle problem
			#since our line segment iz in the direction z only we can simply remove z components to project down into 2d (x,y)
			
			#I will use the technique described by Wolfram MathWorld: https://mathworld.wolfram.com/TriangleInterior.html 
			#If we define one of the corners of the triangle as v0 = v_0
			#and define the vectors that point to the other corners as v1 = v_1 - v_0 and v2 = v_2 - v_0
			v_x = point[0]
			v_y = point[1]
			v0_x = face[0][0]
			v0_y = face[0][1]
			v1_x = face[1][0] - v0_x
			v1_y = face[1][1] - v0_y
			v2_x = face[2][0] - v0_x
			v2_y = face[2][1] - v0_y
			#then we can descibe the point v = v0 + a*v1 + b*v2
			#then a = (det(v*v2) - det(v0*v2)) / det(v1*v2) and b = - (det(v*v1) - det(v0*v1)) / det(v1*v2)
			#where det(u*v) = uXv = u_x * v_y - u_y*v_x
			det_v_v2 = v_x * v2_y - v_y * v2_x
			det_v0_v2 = v0_x * v2_y - v0_y * v2_x
			det_v1_v2 = v1_x * v2_y - v1_y * v2_x
			
			#don't divide by zero
			if det_v1_v2 != 0:
				a = (det_v_v2 - det_v0_v2) / det_v1_v2
			
				det_v_v1 = v_x * v1_y - v_y * v1_x
				det_v0_v1 = v0_x * v1_y - v0_y * v1_x
			
				b = - (det_v_v1 - det_v0_v1) / det_v1_v2
				if a > 0 and b > 0 and a + b < 1:
					results[ndx] = 1
	return np.sum(results)

#volume of a tetrahedron
@cuda.jit(device = True)
def volume(a, b, c, d):
	#from wikipedia: https://en.wikipedia.org/wiki/Tetrahedron
	a_d_x = a[0] - d[0]
	a_d_y = a[1] - d[1]
	a_d_z = a[2] - d[2]
	
	b_d_x = b[0] - d[0]
	b_d_y = b[1] - d[1]
	b_d_z = b[2] - d[2]
	
	c_d_x = c[0] - d[0]
	c_d_y = c[1] - d[1]
	c_d_z = c[2] - d[2]
	
	cross_x = b_d_y * c_d_z - b_d_z * c_d_y
	cross_y = b_d_z * c_d_x - b_d_x * c_d_z
	cross_z = b_d_x * c_d_y - b_d_y * c_d_x
	
	dot = a_d_x * cross_x + a_d_y * cross_y + a_d_z * cross_z
	
	#don't do abs
	return dot / 6
	
#there are two parts to this problem
#first: determine if the line segment crosses the plane containing the triangle
#then solve the point in triangle problem
@cuda.jit
def intersections_z_kernel(point, z_limt, faces, results):
	#this is a 1D thread
	ndx = cuda.grid(1)
	#the particular face for this thread
	face = faces[ndx]
	
	# -- STEP 1 -- 
	#determine if the line segment crosses the plane containing the triangle
	#got this from stack overflow: https://stackoverflow.com/questions/53962225/how-to-know-if-a-line-segment-intersects-a-triangle-in-3d-space
	a = face[0]
	b = face[1]
	c = face[2]
	d = point
	e = (point[0], point[1], z_limt)
	Td = volume(a, b, c, d)
	Te = volume(a, b, c, e)
	
	#if Td and Te have the same sign then the points lie on the same side of the plane
	#if they have opposite sign then they cross the plane
	#if one of them is zero that point lies on the plane
	#so this should be simply:
	if Td*Te <= 0.0:
		# -- STEP 2 --
		#solve the point in triangle problem
		#since our line segment iz in the direction z only we can simply remove z components to project down into 2d (x,y)
		
		#I will use the technique described by Wolfram MathWorld: https://mathworld.wolfram.com/TriangleInterior.html 
		#If we define one of the corners of the triangle as v0 = v_0
		#and define the vectors that point to the other corners as v1 = v_1 - v_0 and v2 = v_2 - v_0
		v_x = point[0]
		v_y = point[1]
		v0_x = a[0]
		v0_y = a[1]
		v1_x = b[0] - v0_x
		v1_y = b[1] - v0_y
		v2_x = c[0] - v0_x
		v2_y = c[1] - v0_y
		#then we can descibe the point v = v0 + a*v1 + b*v2
		#then a = (det(v*v2) - det(v0*v2)) / det(v1*v2) and b = - (det(v*v1) - det(v0*v1)) / det(v1*v2)
		#where det(u*v) = uXv = u_x * v_y - u_y*v_x
		det_v_v2 = v_x * v2_y - v_y * v2_x
		det_v0_v2 = v0_x * v2_y - v0_y * v2_x
		det_v1_v2 = v1_x * v2_y - v1_y * v2_x
			
		#don't divide by zero
		if det_v1_v2 != 0:
			a = (det_v_v2 - det_v0_v2) / det_v1_v2
		
			det_v_v1 = v_x * v1_y - v_y * v1_x
			det_v0_v1 = v0_x * v1_y - v0_y * v1_x
			
			b = - (det_v_v1 - det_v0_v1) / det_v1_v2
			if a > 0 and b > 0 and a + b < 1:
				results[ndx] = 1.0
	#just for testing
	#results[ndx] = Td*Te

@cuda.jit
def intersections_z_kernel_alt(point, z_limt, faces, results):
	#this is a 1D thread
	ndx = cuda.grid(1)
	#the particular face for this thread
	face = faces[ndx]
	
	# -- STEP 1 -- 
	#determine if the line segment crosses the plane containing the triangle
	#got this from stack overflow: https://stackoverflow.com/questions/53962225/how-to-know-if-a-line-segment-intersects-a-triangle-in-3d-space
	a = face[0]
	b = face[1]
	c = face[2]
	d = point
	e = (point[0], point[1], z_limt)
	Td = volume(a, b, c, d)
	Te = volume(a, b, c, e)
	
	#if Td and Te have the same sign then the points lie on the same side of the plane
	#if they have opposite sign then they cross the plane
	#if one of them is zero that point lies on the plane
	#so this should be simply:
	if Td*Te <= 0.0:
		#-- STEP 2 --
		#unlike the functions above we will use the technique given as example by the stack overflow page instead of wolfram
		#because we already have the tetrahedron volume equation
		T_abde = volume(a, b, d, e)
		T_bcde = volume(b, c, d, e)
		T_cade = volume(c, a, d, e)
		
		#if these have the same sign then the segment de intersect the triangle
		if T_abde >= 0.0 and T_bcde >= 0.0 and T_cade >= 0.0:
			results[ndx] = 1.0
		elif T_abde < 0.0 and T_bcde < 0.0 and T_cade < 0.0:
			results[ndx] = 1.0

# -- USE THESE FUNCTIONS --

#This function take an array of faces and a point
#and returns the number of faces that a projection of that point, in the z direction, intersects
def intersections_z(point, faces):
	#copies neccessary
	this_point = np.array(point, dtype = np.float32)
	these_faces = np.ascontiguousarray(faces)
	results = np.zeros(these_faces.shape[0])
	blocks_per_grid = math.ceil(these_faces.shape[0] / THREADS_PER_BLOCK)
	z_limt = np.amin(faces[:,2])
	
	intersections_z_kernel[blocks_per_grid, THREADS_PER_BLOCK](this_point, z_limt, these_faces, results)
	#print(results)
	return np.sum(results)

def intersections_z_alt(point, faces):
	#copies neccessary
	this_point = np.array(point, dtype = np.float32)
	these_faces = np.ascontiguousarray(faces)
	results = np.zeros(these_faces.shape[0])
	blocks_per_grid = math.ceil(these_faces.shape[0] / THREADS_PER_BLOCK)
	z_limt = np.amin(faces[:,2])
	
	intersections_z_kernel_alt[blocks_per_grid, THREADS_PER_BLOCK](this_point, z_limt, these_faces, results)
	#print(results)
	return np.sum(results)

