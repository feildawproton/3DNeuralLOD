#ifndef _COMPUTE_INSIDE_H_
#define _COMPUTE_INSIDE_H_

const unsigned THREADS_PER_BLOCK = 256;
const unsigned CPU_THREADS = 8;

struct vec3
{
    float x, y, z;
};
typedef struct vec3 vec3;

struct triangle
{
    vec3 a;  //have to include struct namespace
    vec3 b;
    vec3 c;
};
typedef struct triangle triangle;

struct vec2
{
    float x;
    float y;
};
typedef struct vec2 vec2;

//this allocates memory.  caller needs to free
//inputs should be pointers to cpu memory
//find if the points are inside the mesh (this assumes the mesh doesn't have holes)
int* create_inside_z_results(const vec3* p_points, const unsigned n_points, const triangle* p_faces, const unsigned n_faces);

#endif // !COMPUTE_INSIDE_H_
