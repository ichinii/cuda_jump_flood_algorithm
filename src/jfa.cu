#include "jfa.h"

#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

using namespace glm;

constexpr const int invalid_pointer = -1;

__device__
int coord_to_id(ivec2 coord, int w) {
    return coord.x + coord.y*w;
}

__device__
ivec2 id_to_coord(int id, int w) {
    return ivec2(id % w, id / w);
}

__device__
bool is_coord_in_bounds(ivec2 coord, int w) {
    return 0 <= coord.x && 0 <= coord.y && coord.x < w && coord.y < w;
}

__device__
bool is_id_in_bounds(int id, int w) {
    return 0 <= id && id < w*w;
}

__device__
ivec2 wrap_coord_clamp(ivec2 coord, int w) {
    return clamp(coord, ivec2(0), ivec2(w-1));
}

__device__
ivec2 wrap_coord_repeat(ivec2 coord, int w) {
    return (coord + ivec2(w)) % ivec2(w);
}

__global__
void jfa_impl_0(int *pointers, int w, int s) {
    int gid = blockIdx.x*blockDim.x + threadIdx.x;

    ivec2 tcoord = id_to_coord(gid, w);
    int tpid = pointers[gid]; // loading early hides latency
    float tl = w*w*2;

    for (int y = -1; y < 2; ++y) {
        for (int x = -1; x < 2; ++x) {
            // if (x == 0 && y == 0)
            //     continue;

            // ivec2 coord = wrap_coord_repeat(tcoord + ivec2(x, y) * s, w);
            ivec2 coord = tcoord + ivec2(x, y) * s;
            if (!is_coord_in_bounds(coord, w))
                continue;

            // ivec2 coord = id_to_coord((gid + y*w*s + x*s) % (w*w), w);
            int id = coord_to_id(coord, w);
            int pid = pointers[id];

            if (pid != invalid_pointer) {
                ivec2 pcoord = id_to_coord(pid, w);
                auto v = pcoord - tcoord;
                float l = (v.x*v.x + v.y*v.y);

                if (tpid == invalid_pointer || l < tl) {
                    tpid = pid;
                    tl = l;
                }
            }
        }
    }

    pointers[gid] = tpid;
}

void jfa_0(unsigned int B, unsigned int T, int *pointers, int w) {
    int s = w/2;
    while (s > 0) {
        jfa_impl_0<<<B, T>>>(pointers, w, s);
        s /= 2;
    }
}

template <bool X, bool Y>
__global__
void jfa_impl_1(int *pointers, int w, int s) {
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ int pids_0[128];
    __shared__ int pids_1[128];
    __shared__ int pids_2[128];

    pids_0[tid] = pointers[gid];

    int id_1 = gid - X*s - Y*s*w;
    int id_2 = gid + X*s + Y*s*w;

    if (is_id_in_bounds(id_1, w)) {
        pids_1[tid] = pointers[id_1];
    }
    if (is_id_in_bounds(id_2, w))
        pids_2[tid] = pointers[id_2];

    __syncthreads();

    ivec2 tcoord = id_to_coord(gid, w);
    float tl = w*w*2;
    int tpid = invalid_pointer;

    {
        int pid = pids_0[tid];
        if (pid != invalid_pointer) {
            ivec2 pcoord = id_to_coord(pid, w);
            auto v = pcoord - tcoord;
            float l = (v.x*v.x + v.y*v.y);

            if (l < tl) {
                tpid = pid;
                tl = l;
            }
        }
    }

    if (is_id_in_bounds(id_1, w)) {
        int pid = pids_1[tid];
        if (pid != invalid_pointer) {
            ivec2 pcoord = id_to_coord(pid, w);
            auto v = pcoord - tcoord;
            float l = (v.x*v.x + v.y*v.y);

            if (l < tl) {
                tpid = pid;
                tl = l;
            }
        }
    }

    if (is_id_in_bounds(id_2, w)) {
        int pid = pids_2[tid];
        if (pid != invalid_pointer) {
            ivec2 pcoord = id_to_coord(pid, w);
            auto v = pcoord - tcoord;
            float l = (v.x*v.x + v.y*v.y);

            if (l < tl) {
                tpid = pid;
                tl = l;
            }
        }
    }

    pointers[gid] = tpid;
}

void jfa_1(unsigned int B, unsigned int T, int *pointers, int w) {
    int s = w/2;
    while (s > 0) {
        jfa_impl_1<1, 0><<<B, T>>>(pointers, w, s);
        jfa_impl_1<0, 1><<<B, T>>>(pointers, w, s);
        // jfa_impl_1<<<B, T>>>(pointers, w, s);
        s /= 2;
    }
}

__device__ unsigned int encode(int id, int l) {
    return (l << 16) | (id & (1<<16));
}

__device__
void jfa_warp(volatile int *pids, int tid, int s) {
    int tpid = pids[tid];
    int pid  = pids[tid + s];
    int tl = tpid - tid;
    int l = pid - tid;

    unsigned int a = encode(tpid, tl);
    unsigned int b = encode(pid, l);
    unsigned int r = min(a, b);
    pids[tid] = r;
    __syncwarp();
}

__global__
void jfa_impl_2(int *pointers) {
    int tid = threadIdx.x;
    int gid = blockIdx.x*blockDim.x + threadIdx.x;

    extern __shared__ int pids[];
    pids[tid] = pointers[gid];
    pids[tid + blockDim.x] = pointers[gid + blockDim.x];

    __syncthreads();

    for (int s = 32; 0 < s; s /= 2) {
        jfa_warp(pids, tid, s);
    }

    pointers[gid] = pids[tid];
}

void jfa_2(unsigned int B, unsigned int T, int *pointers, int w) {
    // int s = w/2;
    // while (s > 0) {
        jfa_impl_2<<<1, T, T*4*2>>>(pointers);
        // s /= 2;
    // }
}

__global__
extern void jfa_init_pointers(int *pointers, bool *input, int w) {
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    pointers[gid] = input[gid] ? gid : invalid_pointer;
}

__global__
void jfa_to_sdf(float *sdf, int *pointers, int w) {
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    auto a = vec2(id_to_coord(gid, w));
    auto b = vec2(id_to_coord(pointers[gid], w));
    // sdf[gid] = length(b - a);
    sdf[gid] = length(b - a) / (w * 2.0f * sqrt(2.0f));
}
