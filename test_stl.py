import os
import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from matplotlib.pylab import cm
from matplotlib.colors import ListedColormap

from sample_mesh import sample_mesh, sample_mesh_selective

if __name__ == "__main__":

	figure = plt.figure()
	axes = mplot3d.Axes3D(figure)

	the_mesh = mesh.Mesh.from_file(os.path.join("data", "makehumanmale.stl"))
	'''
	axes.add_collection3d(mplot3d.art3d.Poly3DCollection(the_mesh.vectors))

	#have to scale the mesh
	scale = the_mesh.points.flatten()
	axes.auto_scale_xyz(scale, scale, scale)

	plt.show()
	'''

	points, labels = sample_mesh(the_mesh, 10000)
	#points, labels = sample_mesh_selective(the_mesh)

	#transparent colormap from https://riptutorial.com/matplotlib/example/11646/using-custom-colormaps
	cm.register_cmap(name="purple_transparency",
                 data={'red':   [(0.,0.4,0.4),
                                 (1.,0.4,0.4)],

                       'green': [(0.,0.0,0.0),
                                 (1.,0.0,0.0)],

                       'blue':  [(0.,1,1),
                                 (1.,1,1)],

                       'alpha': [(0.,0,0),
                                 (1,1,1)]})

	axes.scatter3D(points[:,0], points[:,1], points[:,2], c=labels, cmap = "purple_transparency")
	#axes.scatter3D(points[:,0], points[:,1], points[:,2], c=labels)
	plt.show()