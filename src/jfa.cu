#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include "jfa.h"

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

template <int X, int Y>
__global__
void jfa_impl_3(int *pointers, int w, int s) {
    int gid = blockIdx.x*blockDim.x + threadIdx.x;

    ivec2 tcoord = id_to_coord(gid, w);
    int tpid = pointers[gid]; // loading early hides latency
    float tl = w*w*2;

    for (int i = -1; i < 2; ++i) {
        ivec2 coord = tcoord + ivec2(i * X, i * Y) * s;
        if (!is_coord_in_bounds(coord, w))
            continue;

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

    pointers[gid] = tpid;
}

template <int X, int Y>
__global__
void jfa_impl_4(int *pointers, int w, int s) {
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    constexpr const int T = 128;

    __shared__ int pids[T * 3];

    pids[tid]       = pointers[X * ((gid - T + w*w) % (w*w)) + Y * (gid - w + w*w) % (w*w)];
    pids[tid + T]   = pointers[gid];
    pids[tid + T*2] = pointers[X * ((gid + T) % (w*w)) + Y * (gid + w) % (w*w)];

    __syncthreads();

    ivec2 tcoord = id_to_coord(gid, w);
    int tpid = pids[tid + T];
    float tl = w*w*2;

    while (0 < s) {
        for (int i = 0; i < 2; ++i) {
            int pid = pids[tid+T - s + s*i*2];

            if (pid != invalid_pointer) {
                ivec2 pcoord = id_to_coord(pid, w);
                auto v = pcoord - tcoord;
                float l = v.x*v.x + v.y*v.y;

                if (tpid == invalid_pointer || l < tl) {
                    tpid = pid;
                    tl = l;
                }
            }
        }

        pids[tid + 128] = tpid;
        s /= 2;
        __syncthreads();
    }

    pointers[gid] = tpid;
}

void jfa_2(unsigned int B, unsigned int T, int *pointers, int w) {
    int s = w/2;
    while (0 < s) {
        jfa_impl_3<1, 0><<<B, T>>>(pointers, w, s);
        s /= 2;
    }
    s = w/2;
    while (0 < s) {
        jfa_impl_3<0, 1><<<B, T>>>(pointers, w, s);
        s /= 2;
    }
}

void jfa_3(unsigned int B, unsigned int T, int *pointers, int w) {
    assert(w % T == 0);
    assert(T == 128);
    int b = w / T;

    int s = w/2;
    while (T < s) {
        jfa_impl_3<1, 0><<<B, T>>>(pointers, w, s);
        s /= 2;
    }
    jfa_impl_4<1, 0><<<B, T>>>(pointers, w, 128);

    s = w/2;
    while (0 < s) {
        jfa_impl_3<0, 1><<<B, T>>>(pointers, w, s);
        s /= 2;
    }
}

void jfa_4(unsigned int B, unsigned int T, int *pointers, int w) {
}
