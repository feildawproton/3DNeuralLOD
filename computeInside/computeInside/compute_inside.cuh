#ifndef COMPUTE_INSIDE_H_
#define COMPUTE_INSIDE_H_

const unsigned THREADS_PER_BLOCK = 256;
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

#endif // !COMPUTE_INSIDE_H_
