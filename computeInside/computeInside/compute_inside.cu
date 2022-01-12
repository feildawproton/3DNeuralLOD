//I implemented this in numba cuda as well
//that worked and parallelized checking intersections for a given points
//however I also wanted to parallelize over the points
//So, parallel foreach point which parallelize foreach triangle
//A way to do this would be to multithread points on CPU which launch GPU kernels to parallelize intersections for each point
//It's difficult to parallelize in python so I fell back to doing this in CUDA C

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <omp.h>    //for parallel
#include <stdlib.h> //for rand
#include <math.h>   //for ceil()

const unsigned THREADS_PER_BLOCK = 64;
const unsigned CPU_THREADS = 8;

typedef struct vec3
{
    float x, y, z;
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
//the results are floats. 1.0 the face was crossed and 0.0 the face wasn't crossed.  could I use a smaller datatype?
__global__ void intersections_z_kernel(const vec3 point, const float z_limt, const triangle* faces, const int n_faces, float* results)
{
    //need to calculate the correct global index
    int ndx = blockIdx.x * blockDim.x + threadIdx.x;
    //don't want to go out of memory
    if (ndx < n_faces)
    {
        results[ndx] = 0.0; //set to zero first
        triangle face = faces[ndx];
        
        // -- STEP 1 --
        //determine if the line segment crossed the plane of the triangle
        //got this from stack overflow: https://stackoverflow.com/questions/53962225/how-to-know-if-a-line-segment-intersects-a-triangle-in-3d-space
        vec3 a      = face.a;
        vec3 b      = face.b;
        vec3 c      = face.c;
        vec3 d      = point;
        vec3 e      = {};
        e.x         = point.x;
        e.y         = point.y;
        e.z         = z_limt;
        float Td    = volume(a, b, c, d);
        float Te    = volume(a, b, c, e);
        
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
            vec2 v  = {};
            v.x     = point.x;
            v.y     = point.y;
            vec2 v0 = {};
            v0.x    = a.x;
            v0.y    = a.y;
            vec2 v1 = {};
            v1.x    = b.x - v0.x;
            v1.y    = b.y - v0.y;
            vec2 v2 = {};
            v2.x    = c.x - v0.x;
            v2.y    = c.y - v0.y;
            //then we can describe the point v = v0 + a*v1 + b*v2
            //the a = (det(v*v2) - det(v0*v2)) / det(v1*v2) and b = - (det(v*v1) - det(v0*v1)) / det(v1*v2)
            //where det(u*v) = uXv = u_x * v_y - u_y * v_x
            //make sure we don't divide by zero
            float det_v1_v2 = v1.x * v2.y - v1.y * v2.x;
            if (det_v1_v2 != 0.0)
            {
                float det_v_v2  = v.x * v2.y - v.y * v2.x;
                float det_v0_v2 = v0.x * v2.y - v0.y * v2.x;
                float a         = (det_v_v2 - det_v0_v2) / det_v1_v2;

                float det_v_v1  = v.x * v1.y - v.y * v1.x;
                float det_v0_v1 = v0.x * v1.y - v0.y * v1.x;
                float b         = - (det_v_v1 - det_v0_v1) / det_v1_v2;

                if (a > 0.0 && b > 0.0 && a + b < 1.0)
                    results[ndx] = 1.0;
            }
        }
    }
}

//This alternate implementation is slower than the one above
//I keep it for reference though because it doesn't require the 2D projection
//in my previous testing (in python numba cuda) these yield the same results.  Perhaps need to test again.
//the results are floats. 1.0 the face was crossed and 0.0 the face wasn't crossed.  could I use a smaller datatype?
//results should be intitialized to all zeros
__global__ void intersections_z_kernel_alt(const vec3 point, const float z_limt, const triangle* faces, const int n_faces, float* results)
{
    //need to calculate the correct global index
    int ndx = blockIdx.x * blockDim.x + threadIdx.x;
    //don't want to go out of memory
    if (ndx < n_faces)
    {
        results[ndx] = 0.0; //set to zero first
        triangle face = faces[ndx];

        // -- STEP 1 --
        //determine if the line segment crossed the plane of the triangle
        //got this from stack overflow: https://stackoverflow.com/questions/53962225/how-to-know-if-a-line-segment-intersects-a-triangle-in-3d-space
        vec3 a      = face.a;
        vec3 b      = face.b;
        vec3 c      = face.c;
        vec3 d      = point;
        vec3 e      = {};
        e.x         = point.x;
        e.y         = point.y;
        e.z         = z_limt;
        float Td    = volume(a, b, c, d);
        float Te    = volume(a, b, c, e);

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
float count_intersections(float* intersections)
{

}
*/

//inputs should be pointers to cpu memory
//find if the points are inside the mesh (this assumes the mesh doesn't have holes)
float inside_z(const vec3* p_points, const unsigned n_points, const triangle* p_faces, const unsigned n_faces)
{
    //select gpu
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaSetDevice failed.");

    printf("allocating gpu memory and copying from host-to-device\n");
    // -- allocate memory --
    triangle* p_faces_gpu;

    cudaStatus = cudaMalloc((void**)&p_faces_gpu, sizeof(triangle) * n_faces);
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaMalloc failed!\n");

    // -- copy memory from host to device --
    cudaStatus = cudaMemcpy(p_faces_gpu, p_faces, sizeof(triangle) * n_faces, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "memory copy host-to-device failed!\n");

    unsigned BLOCKS_PER_GRID = ceil((float)n_faces / (float)THREADS_PER_BLOCK); //caste to float before division to get decimal
    printf("each thread will launch %i blocks per grid with %i threads per block\n", BLOCKS_PER_GRID, THREADS_PER_BLOCK);
    //divide the points accross the threads
    #pragma omp parallel
    {
        printf("launching cpu thread %i of %i\n", omp_get_thread_num(), omp_get_num_threads());
        #pragma omp for
        for(int i = 0; i < n_points; i++)
        {
            float* p_intersect_faces;
            cudaStatus = cudaMalloc((void**)&p_intersect_faces, sizeof(float) * n_faces);
            if (cudaStatus != cudaSuccess)
                fprintf(stderr, "cudaMalloc failed!\n");
            intersections_z_kernel << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> > (p_points[i], 1.0, p_faces_gpu, n_faces, p_intersect_faces);

            //perhaps should free in a separate loop after we are done?
            cudaFree(p_intersect_faces);
        }
    }

    cudaFree(p_faces_gpu);

    return 0.0;
}

/*
void test_parallel()
{
    omp_set_num_threads(CPU_THREADS);
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        printf("hello from thread: %d of %d\n", thread_id, omp_get_num_threads());
    }
}
*/

vec3* alloc_rand_points(unsigned num_points)
{
    vec3* p_points;
    p_points = (vec3*)malloc(sizeof(vec3) * num_points);

    for (unsigned i = 0; i < num_points; i++)
    {
        p_points[i].x     = float(rand()) / (float)RAND_MAX;
        p_points[i].y     = float(rand()) / (float)RAND_MAX;
        p_points[i].z     = float(rand()) / (float)RAND_MAX;
    }
    return p_points;
}

triangle* alloc_rand_faces(unsigned num_faces)
{
    triangle* p_faces;
    p_faces = (triangle*)malloc(sizeof(triangle) * num_faces);

    for (unsigned i = 0; i < num_faces; i++)
    {
        p_faces[i].a.x = float(rand()) / (float)RAND_MAX;
        p_faces[i].a.y = float(rand()) / (float)RAND_MAX;
        p_faces[i].a.z = float(rand()) / (float)RAND_MAX;

        p_faces[i].b.x = float(rand()) / (float)RAND_MAX;
        p_faces[i].b.y = float(rand()) / (float)RAND_MAX;
        p_faces[i].b.z = float(rand()) / (float)RAND_MAX;

        p_faces[i].c.x = float(rand()) / (float)RAND_MAX;
        p_faces[i].c.y = float(rand()) / (float)RAND_MAX;
        p_faces[i].c.z = float(rand()) / (float)RAND_MAX;
    }
    return p_faces;
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    //test_parallel();
    unsigned n_points = 100;
    unsigned n_faces = 100;
    vec3* p_points = alloc_rand_points(n_points);
    triangle* p_faces = alloc_rand_faces(n_faces);
    /*
    for (unsigned i = 0; i < 100; i++)
    {
        printf("%f, %f, %f\n", p_faces[i].a.x, p_faces[i].a.y, p_faces[i].a.z);
        printf("%f, %f, %f\n", p_faces[i].b.x, p_faces[i].b.y, p_faces[i].b.z);
        printf("%f, %f, %f\n", p_faces[i].c.x, p_faces[i].c.y, p_faces[i].c.z);
    }
    */
    float result = inside_z(p_points, n_points, p_faces, n_faces);

    free(p_faces);
    free(p_points);
    return 0;
}

