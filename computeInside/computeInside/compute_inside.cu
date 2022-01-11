//I implemented this in numba cuda as well
//that worked and parallelized checking intersections for a given points
//however I also wanted to parallelize over the points
//So, parallel foreach point which parallelize foreach triangle
//A way to do this would be to multithread points on CPU which launch GPU kernels to parallelize intersections for each point
//It's difficult to parallelize in python so I fell back to doing this in CUDA C

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <omp.h>

typedef struct vec3
{
    float x;
    float y;
    float z;
};

typedef struct triangle
{
    vec3 a;
    vec3 b;
    vec3 c;
};

typedef struct vec2
{
    float x;
    float y;
};

//computes the volume of a tetrahedron
__device__
float volume(vec3 a, vec3 b, vec3 c, vec3 d)
{
    vec3 a_d = {};
    a_d.x = a.x - d.x;
    a_d.y = a.y - d.y;
    a_d.z = a.z - d.z;

    vec3 b_d = {};
    b_d.x = b.x - d.x;
    b_d.y = b.y - d.y;
    b_d.z = b.z - d.z;

    vec3 c_d = {};
    c_d.x = c.x - d.x;
    c_d.y = c.y - d.y;
    c_d.z = c.z - d.z;

    vec3 crss = {};
    crss.x = b_d.y * c_d.z - b_d.z * c_d.y;
    crss.y = b_d.z * c_d.x - b_d.x * c_d.z;
    crss.z = b_d.x * c_d.y - b_d.y * c_d.x;

    float numerator = (a_d.x * crss.x) + (a_d.y * crss.y) + (a_d.z * crss.z);

    return (numerator / 6);
}

//determine which triangles a line segment cross
//there are two parts to this problem
//step 1: determine if the line segment cross the plane the triangle is in
//step 2: determine if the point, compressed along the line segment, is inside the triangle (also projected into 2d along the line segment)
//in this case the line segment will be along zee only.  we have a point and then just project to the zee limit
//the results are floats.  could I use a smaller datatype?
__global__ void intersections_z_kernel(const vec3 point, const float z_limt, const triangle* faces, const int n_faces, float* results)
{
    //need to calculate the correct global index
    int ndx = blockIdx.x * blockDim.x + threadIdx.x;
    //don't want to go out of memory
    if (ndx < n_faces)
    {
        triangle face = faces[ndx];
        
        // -- STEP 1 --
        //determine if the line segment crossed the plane of the triangle
        //got this from stack overflow: https://stackoverflow.com/questions/53962225/how-to-know-if-a-line-segment-intersects-a-triangle-in-3d-space
        vec3 a = face.a;
        vec3 b = face.b;
        vec3 c = face.c;
        vec3 d = point;
        vec3 e = {};
        e.x = point.x;
        e.y = point.y;
        e.z = z_limt;
        float Td = volume(a, b, c, d);
        float Te = volume(a, b, c, e);
        
        //if Td and Te have the same sign then the points lie on one side of the plane
        //if they have opposite signes then it crosses the plane
        //if one of them is zero then that point is on the plane
        //I think this is the simplist way to check:
        if (Td * Te <= 0.0)
        {
            // -- STEP 2 --
            //solve the point in triangle problem
            //since we define the line segment along z only
            //projecting to 2d is simply removing the z components

            //the method I use here is from Wolfram Mathworld: https://mathworld.wolfram.com/TriangleInterior.html 
            //if we define one of the corners of the triangle as v0 = v_0
            //and define the vectors that point to the other corners as v1 = v_1 - v_0 and v2 = v_2 - v_0
            vec2 v = {};
            v.x = point.x;
            v.y = point.y;
            vec2 v0 = {};
            v0.x = a.x;
            v0.y = a.y;
            vec2 v1 = {};
            v1.x = b.x - v0.x;
            v1.y = b.y - v0.y;
            vec2 v2 = {};
            v2.x = c.x - v0.x;
            v2.y = c.y - v0.y;
            //then we can describe the point v = v0 + a*v1 + b*v2
            //the a = (det(v*v2) - det(v0*v2)) / det(v1*v2) and b = - (det(v*v1) - det(v0*v1)) / det(v1*v2)
            //where det(u*v) = uXv = u_x * v_y - u_y * v_x
            //make sure we don't divide by zero
            float det_v1_v2 = v1.x * v2.y - v1.y * v2.x;
            if (det_v1_v2 != 0.0)
            {
                float det_v_v2 = v.x * v2.y - v.y * v2.x;
                float det_v0_v2 = v0.x * v2.y - v0.y * v2.x;
                float a = (det_v_v2 - det_v0_v2) / det_v1_v2;

                float det_v_v1 = v.x * v1.y - v.y * v1.x;
                float det_v0_v1 = v0.x * v1.y - v0.y * v1.x;
                float b = - (det_v_v1 - det_v0_v1) / det_v1_v2;

                if (a > 0.0 && b > 0.0 && a + b < 1.0)
                    results[ndx] = 1.0;
            }
        }
    }
}

//This alternate implementation is slower than the one above
//I keep it for reference though because it doesn't require the 2D projection
//in my previous testing (in python numba cuda) these yield the same results.  Perhaps need to test again.
__global__ void intersections_z_kernel_alt(const vec3 point, const float z_limt, const triangle* faces, const int n_faces, float* results)
{
    //need to calculate the correct global index
    int ndx = blockIdx.x * blockDim.x + threadIdx.x;
    //don't want to go out of memory
    if (ndx < n_faces)
    {
        triangle face = faces[ndx];

        // -- STEP 1 --
        //determine if the line segment crossed the plane of the triangle
        //got this from stack overflow: https://stackoverflow.com/questions/53962225/how-to-know-if-a-line-segment-intersects-a-triangle-in-3d-space
        vec3 a = face.a;
        vec3 b = face.b;
        vec3 c = face.c;
        vec3 d = point;
        vec3 e = {};
        e.x = point.x;
        e.y = point.y;
        e.z = z_limt;
        float Td = volume(a, b, c, d);
        float Te = volume(a, b, c, e);

        //if Td and Te have the same sign then the points lie on one side of the plane
        //if they have opposite signes then it crosses the plane
        //if one of them is zero then that point is on the plane
        //I think this is the simplist way to check:
        if (Td * Te <= 0.0)
        {
            // -- STEP 2 --
            //so I continue on with the method described in the stack overflow help above
            float T_abde = volume(a, b, d, e);
            float T_bcde = volume(b, c, d, e);
            float T_cade = volume(c, a, d, e);

            //if these have the same sine then the segment does intersect the triangle
            if (T_abde >= 0.0 && T_bcde >= 0.0 && T_cade >= 0.0)
                results[ndx] = 1.0;
            else if (T_abde < 0.0 && T_bcde < 0.0 && T_cade < 0.0)
                results[ndx] = 1.0;
        }
    }
}

/*
float intersections_z(const vec3 point, const triangle* faces, const int n_faces)
{

}
*/

void test_parallel()
{
    printf("Using %d threads\n", omp_get_num_threads());
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        printf("hello from thread: %d\n", thread_id);
    }
}

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    omp_set_num_threads(8);
    test_parallel();

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
