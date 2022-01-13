//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include <stdlib.h> //for rand and malloc
#include <stdio.h>  //for printing

#include "compute_inside.cuh"

vec3* create_rand_points(unsigned num_points)
{
    vec3* p_points;
    p_points = (vec3*)malloc(sizeof(vec3) * num_points);

    for (unsigned i = 0; i < num_points; i++)
    {
        p_points[i].x = (float)rand() / (float)RAND_MAX;
        p_points[i].y = (float)rand() / (float)RAND_MAX;
        p_points[i].z = (float)rand() / (float)RAND_MAX;
    }
    return p_points;
}

triangle* create_rand_faces(unsigned num_faces)
{
    triangle* p_faces;
    p_faces = (triangle*)malloc(sizeof(triangle) * num_faces);

    for (unsigned i = 0; i < num_faces; i++)
    {
        p_faces[i].a.x = (float)rand() / (float)RAND_MAX;
        p_faces[i].a.y = (float)rand() / (float)RAND_MAX;
        p_faces[i].a.z = (float)rand() / (float)RAND_MAX;

        p_faces[i].b.x = (float)rand() / (float)RAND_MAX;
        p_faces[i].b.y = (float)rand() / (float)RAND_MAX;
        p_faces[i].b.z = (float)rand() / (float)RAND_MAX;

        p_faces[i].c.x = (float)rand() / (float)RAND_MAX;
        p_faces[i].c.y = (float)rand() / (float)RAND_MAX;
        p_faces[i].c.z = (float)rand() / (float)RAND_MAX;
    }
    return p_faces;
}

int main()
{

    unsigned n_points = 10000;
    unsigned n_faces = 10000;

    vec3* p_points = create_rand_points(n_points);
    triangle* p_faces = create_rand_faces(n_faces);

    /*
    for (unsigned i = 0; i < 100; i++)
    {
        printf("%f, %f, %f\n", p_faces[i].a.x, p_faces[i].a.y, p_faces[i].a.z);
        printf("%f, %f, %f\n", p_faces[i].b.x, p_faces[i].b.y, p_faces[i].b.z);
        printf("%f, %f, %f\n", p_faces[i].c.x, p_faces[i].c.y, p_faces[i].c.z);
    }
    */
    int* p_results = create_inside_z_results(p_points, n_points, p_faces, n_faces);

    /*
    for (unsigned i = 0; i < n_points; i++)
    {
        printf("%i\n", p_results[i]);
    }
    */

    free(p_results);
    free(p_faces);
    free(p_points);


    return 0;
}