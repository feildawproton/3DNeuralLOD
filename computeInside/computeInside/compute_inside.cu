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
#include <stdlib.h> //for rand and malloc
#include <math.h>   //for ceil()
#include <time.h>   //for perf testing

#include "compute_inside.h"

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
__global__ void intersections_z_kernel(const vec3 point, const float z_limt, const triangle* faces, const int n_faces, int* results)
{
    //need to calculate the correct global index
    int ndx = blockIdx.x * blockDim.x + threadIdx.x;
    //don't want to go out of memory
    if (ndx < n_faces)
    {
        results[ndx] = 0; //set to zero first
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
                    results[ndx] = 1;
            }
        }
    }
}

//This alternate implementation is slower than the one above (on my computer at least)
//I keep it for reference though because it doesn't require the 2D projection
//in my previous testing (in python numba cuda) these yield the same results.  Perhaps need to test again.
//the results are floats. 1.0 the face was crossed and 0.0 the face wasn't crossed.  could I use a smaller datatype?
//results should be intitialized to all zeros
__global__ void intersections_z_kernel_alt(const vec3 point, const float z_limt, const triangle* faces, const unsigned n_faces, int* results)
{
    //need to calculate the correct global index
    int ndx = blockIdx.x * blockDim.x + threadIdx.x;
    //don't want to go out of memory
    if (ndx < n_faces)
    {
        results[ndx] = 0; //set to zero first
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
                results[ndx] = 1;
            else if (T_abde < 0.0 && T_bcde < 0.0 && T_cade < 0.0)
                results[ndx] = 1;
        }
    }
}

//counts the number of intersection and returns 1 if odd and 0 if even
//1 if inside and 0 if outside of mesh
int count_inside(const int* intersections, const unsigned n_intersections)
{
    int sum = 0;
    for (unsigned i = 0; i < n_intersections; i++)
        sum += intersections[i];
    return (sum % 2);
}

// -- USE THIS FUNCITON -- 
//inputs should be pointers to cpu memory
//find if the points are inside the mesh (this assumes the mesh doesn't have holes)
int* create_inside_z_results(const vec3* p_points, const unsigned n_points, const triangle* p_faces, const unsigned n_faces)
{
    //select gpu
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaSetDevice failed.");

   
    // -- allocate memory --
    int* p_results = (int*)malloc(sizeof(int) * n_points);

    printf("allocating gpu memory and copying from host-to-device\n");
    triangle* p_faces_gpu;
    cudaStatus = cudaMalloc((void**)&p_faces_gpu, sizeof(triangle) * n_faces);
    cudaStatus = cudaMemcpy(p_faces_gpu, p_faces, sizeof(triangle) * n_faces, cudaMemcpyHostToDevice);

    unsigned BLOCKS_PER_GRID = ceil((float)n_faces / (float)THREADS_PER_BLOCK); //caste to float before division to get decimal

    printf("each thread will launch %i blocks per grid with %i threads per block\n", BLOCKS_PER_GRID, THREADS_PER_BLOCK);
    //divide the points accross the threads
    
    //the multiparallel version is faster than just the face parallel (single thread) one is (by about 2x)
    clock_t start = clock();
    #pragma omp parallel
    {
        printf("launching cpu thread %i of %i\n", omp_get_thread_num(), omp_get_num_threads());
        #pragma omp for
        for (int i = 0; i < n_points; i++)
        {
            int* p_intersect_gpu;
            cudaStatus = cudaMalloc((void**)&p_intersect_gpu, sizeof(int) * n_faces);

            //from this we get what faces were intersected
            intersections_z_kernel <<< BLOCKS_PER_GRID, THREADS_PER_BLOCK >>> (p_points[i], 1.0, p_faces_gpu, n_faces, p_intersect_gpu);

            //copy memory back to cpu (could do reduce sum on gpu but I don't)
            int* p_intersect = (int*)malloc(sizeof(int) * n_faces);
            cudaStatus = cudaMemcpy(p_intersect, p_intersect_gpu, sizeof(int) * n_faces, cudaMemcpyDeviceToHost);

            //now we need to sum up the face intersections to get how many faces a givin point intersects
            int inside = count_inside(p_intersect, n_faces);
            p_results[i] = inside;

            //perhaps should free in a separate loop after we are done?
            free(p_intersect);

            cudaFree(p_intersect_gpu);
        }
    }
    cudaDeviceSynchronize();
    clock_t delta = clock() - start;
    float time = delta / CLOCKS_PER_SEC;
    printf("multiparallel (cpu and gpu) mesh sampling execution time: %f\n", time);
    
    cudaFree(p_faces_gpu);
    
    return p_results;
}

/*
vec3* create_rand_points(unsigned num_points)
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

triangle* create_rand_faces(unsigned num_faces)
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
    unsigned n_points = 10000;
    unsigned n_faces = 10000;

    vec3* p_points = create_rand_points(n_points);
    triangle* p_faces = create_rand_faces(n_faces);
    

    //for (unsigned i = 0; i < 100; i++)
    //{
    //    printf("%f, %f, %f\n", p_faces[i].a.x, p_faces[i].a.y, p_faces[i].a.z);
    //    printf("%f, %f, %f\n", p_faces[i].b.x, p_faces[i].b.y, p_faces[i].b.z);
    //    printf("%f, %f, %f\n", p_faces[i].c.x, p_faces[i].c.y, p_faces[i].c.z);
    //}

    int* p_results = create_inside_z_results(p_points, n_points, p_faces, n_faces);

    
    //for (unsigned i = 0; i < n_points; i++)
    //{
    //    printf("%i\n", p_results[i]);
    //}
    

    free(p_results);
    free(p_faces);
    free(p_points);


    return 0;
}
*/

