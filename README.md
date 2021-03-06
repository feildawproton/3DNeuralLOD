# 3DNeuralLOD
Testing Neural LOD in 3D

This is a 3D version of https://github.com/feildawproton/NeuralLOD including some portions written in C and CUDA.

**Mesh sampling**
- Mesh sampling is done with two forms of parallelism
- CPU threading, with openmp, over the sample points
- and GPU kernels for each sample point to parallelize testing the mesh for line segment intersections
- Enabling this in VS requires VS passing the /openmp argument to nvcc
  - in vs19 this is done by going to project Properties -> CUDA C/C++ -> Host -> Additional Compiler Options
  - add /openmp in the text entry box

Sampling blender.org's Suzanne (samples outside of mesh are transparent):
![suzanne_sampled](https://user-images.githubusercontent.com/56926839/149004660-7ae0fe37-4093-47f8-8910-81b8e30183e8.png)

Sampling makehumancommunity.org's default male (samples outside of mesh are transparent):
![makehumanmale_sample_100000](https://user-images.githubusercontent.com/56926839/149004827-0be5bedf-4e5e-4fdc-a978-a866419644e0.png)


