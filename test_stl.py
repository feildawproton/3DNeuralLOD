import os
import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot

if __name__ == "__main__":

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

	sample_mesh(the_mesh, 1000)
