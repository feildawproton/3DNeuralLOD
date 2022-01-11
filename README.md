# 3DNeuralLOD
Testing Neural LOD in 3D

Requirements
- Numba CUDA for mesh sampling in parallel
- Numpy-stl for meshes
![suzanne_sampled](https://user-images.githubusercontent.com/56926839/149004660-7ae0fe37-4093-47f8-8910-81b8e30183e8.png)

**Mesh sampling**
- Mesh sampling is done with two forms of parallelism
- CPU threading, with openmp, over the samples
- and GPU kernels for each sample to parallelize testing the mesh for intersections
- Enabling this in VS requires VS passing the /openmp argument to nvcc
  - in vs19 this is done by going to project Properties -> CUDA C/C++ -> Host -> Additional Compiler Options
  - add /openmp in the test entry box

Attribution:
- The suzanne model is the work of the folks at blender.org
