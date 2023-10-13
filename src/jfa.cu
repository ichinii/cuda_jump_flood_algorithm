#include <iostream>

#include <cuda.h>
#include <glm/glm.hpp>

#include "jfa.h"
#include "misc.h"

using namespace glm;

constexpr const int invalid_pointer = 1<<30;

// constexpr const IdVec2 invalid_pointer2D = IdVec2(-1);
#define invalid_pointer2D IdVec2(-1)

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

__device__
__forceinline__
int transpose_id(int id, int w) {
    if (id == invalid_pointer)
        return invalid_pointer;

    ivec2 coord = id_to_coord(id, w);
    return coord_to_id(ivec2(coord.y, coord.x), w);
}

__global__
void transpose_pointers(int *dst, int *src, int w) {
    unsigned int o0 = blockIdx.x * 32 + blockIdx.y * 32 * w;
    unsigned int o1 = blockIdx.y * 32 + blockIdx.x * 32 * w;

    __shared__ int data[32*32];

    for (int y = 0; y < 32; y += blockDim.y) {
        int pi = threadIdx.x + (y + threadIdx.y) * w;
        int di = threadIdx.y + y + threadIdx.x * 32;

        int p = src[pi + o0];
        data[di] = transpose_id(p, w);
    }

    __syncthreads();

    for (int y = 0; y < 32; y += blockDim.y) {
        int pi = threadIdx.x + (y + threadIdx.y) * w;
        int di = threadIdx.x + (y + threadIdx.y) * 32;

        dst[pi + o1] = data[di];
    }
}

__global__
void transpose_pointers_naive(int *dst, int *src, int w) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int dst_id = coord_to_id(ivec2(x, y), w);
    int src_id = coord_to_id(ivec2(y, x), w);
    int p = src[src_id];
    dst[dst_id] = transpose_id(p, w);
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
            ivec2 coord = tcoord + ivec2(x, y) * s;
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
    int tpid = pointers[gid];
    float tl = w*w*2;

    for (int i = -1; i < 2; ++i) {
        ivec2 coord = tcoord + ivec2(i * X, i * Y) * s;
        if (!is_coord_in_bounds(coord, w))
            continue;

        int id = coord_to_id(coord, w);
        int pid = pointers[id];

        ivec2 pcoord = id_to_coord(pid, w);
        vec2 v = pcoord - tcoord;
        float l = length_squared(v);

        if (l < tl) {
            tpid = pid;
            tl = l;
        }
    }

    pointers[gid] = tpid;
}

template <int X, int Y>
__global__
void jfa_impl_5(IdVec2 *pointers, int w, int16 s) {
    int gid = blockIdx.x*blockDim.x + threadIdx.x;

    IdVec2 tcoord = IdVec2(id_to_coord(gid, w));
    IdVec2 tpid = pointers[gid];
    float tl = w*w*2;

    for (int i = -1; i < 2; ++i) {
        ivec2 coord = tcoord + IdVec2(i * X, i * Y) * s;
        if (!is_coord_in_bounds(coord, w))
            continue;

        IdVec2 pid = pointers[coord_to_id(coord, w)];

        if (pid != invalid_pointer2D) {
            auto v = pid - tcoord;
            float l = (v.x*v.x + v.y*v.y);

            if (tpid == invalid_pointer2D || l < tl) {
                tpid = pid;
                tl = l;
            }
        }
    }

    pointers[gid] = tpid;
}

__device__
__forceinline__
void jfa_impl_4_horizontal_check(int *pids, ivec2 tcoord, int &tpid, float &tl, int w, int s, int blockSize) {
    int tid = threadIdx.x;

    int pid = pids[tid + blockSize + s];

    ivec2 pcoord = id_to_coord(pid, w);
    vec2 v = pcoord - tcoord;
    float l = length_squared(v);

    if (l < tl) {
        tpid = pid;
        tl = l;
    }
}

__device__
__forceinline__
void jfa_impl_4_horizontal_do(int *pids, ivec2 tcoord, int &tpid, float &tl, int w, int s, int blockSize) {
    int tid = threadIdx.x;
    jfa_impl_4_horizontal_check(pids, tcoord, tpid, tl, w, -s, blockSize);
    jfa_impl_4_horizontal_check(pids, tcoord, tpid, tl, w,  s, blockSize);
    pids[tid + blockSize] = tpid;
    __syncthreads();
}

template <int blockSize>
__global__
void jfa_impl_4_horizontal(int *pointers, int w) {
    int gid = blockIdx.x*blockSize + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ int pids[blockSize * 3];

    pids[tid]               = pointers[(gid - blockSize + w*w) % (w*w)];
    pids[tid + blockSize]   = pointers[gid];
    pids[tid + blockSize*2] = pointers[(gid + blockSize) % (w*w)];

    __syncthreads();

    ivec2 tcoord = id_to_coord(gid, w);
    int tpid = pids[tid + blockSize];
    ivec2 pcoord = id_to_coord(tpid, w);
    vec2 v = pcoord - tcoord;
    float l = length_squared(v);
    float tl = l;

#define STEP(T) \
    if constexpr (blockSize >= T) { \
        jfa_impl_4_horizontal_do(pids, tcoord, tpid, tl, w, T, blockSize); \
    }

    STEP(1024)
    STEP( 512)
    STEP( 256)
    STEP( 128)
    STEP(  64)
    STEP(  32)
    STEP(  16)
    STEP(   8)
    STEP(   4)
    STEP(   2)
    STEP(   1)
#undef STEP

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

    int s = w/2;
    while (T < s) {
        std::cout << "jfa_impl_3<1, 0> s=" << s << " ms=" << perf([&] {
            jfa_impl_3<1, 0><<<B, T>>>(pointers, w, s);
        }) << std::endl;
        s /= 2;
    }
    std::cout << "jfa_impl_4_horizontal s=" << s << " ms=" << perf([&] {
        jfa_impl_4_horizontal<128><<<B, T>>>(pointers, w);
    }) << std::endl;

    float ps = 0.0f;
    s = w/2;
    while (0 < s) {
        float p = perf([&] {
            jfa_impl_3<0, 1><<<B, T>>>(pointers, w, s);
        });
        std::cout << "jfa_impl_3<0, 1> s=" << s << " ms=" << p << std::endl;
        ps += p;
        s /= 2;
    }
    std::cout << "ps: " << ps << std::endl;
}

void jfa_4(unsigned int B, unsigned int T, int *pointers, int w) {
    assert(w % T == 0);
    assert(T == 128);

    float perfsum = 0.0f;

    int s = w/2;
    while (T <= s) {
        auto p = perf([&] {
            jfa_impl_3<1, 0><<<B, T>>>(pointers, w, s);
        });
        perfsum += p;
        std::cout << "jfa_impl_3<1, 0> s=" << s << " ms=" << p << std::endl;
        s /= 2;
    }
    auto p = perf([&] {
        jfa_impl_4_horizontal<128><<<B, T>>>(pointers, w);
    });
    perfsum += p;
    std::cout << "jfa_impl_4_horizontal s=" << s << " ms=" << p << std::endl;

    int *pt;
    cudaMalloc(&pt, w*w * sizeof(int));
    p = perf([&] {
        auto dimBlock = dim3(32, 4);
        auto dimGrid = dim3(w / 32, w / 32);
        transpose_pointers<<<dimGrid, dimBlock>>>(pt, pointers, w);
    });
    std::swap(pt, pointers);
    perfsum += p;
    std::cout << "transpose_pointers ms=" << p << std::endl;

    s = w/2;
    while (T <= s) {
        auto p = perf([&] {
            jfa_impl_3<1, 0><<<B, T>>>(pointers, w, s);
        });
        perfsum += p;
        std::cout << "jfa_impl_3<1, 0> s=" << s << " ms=" << p << std::endl;
        s /= 2;
    }
    p = perf([&] {
        jfa_impl_4_horizontal<128><<<B, T>>>(pointers, w);
    });
    perfsum += p;
    std::cout << "jfa_impl_4_horizontal s=" << s << " ms=" << p << std::endl;

    std::cout << perfsum << std::endl;

    p = perf([&] {
        auto dimBlock = dim3(32, 4);
        auto dimGrid = dim3(w / 32, w / 32);
        transpose_pointers<<<dimGrid, dimBlock>>>(pt, pointers, w);
    });
    std::swap(pt, pointers);
    cudaFree(pt);
    perfsum += p;
    std::cout << "transpose_pointers ms=" << p << std::endl;

    CHECK_LAST_CUDA_ERROR();
}

__global__
void jfa_init_pointers_2D(IdVec2 *pointers, bool *input, int w) {
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    pointers[gid] = input[gid] ? IdVec2(gid % w, gid / w) : invalid_pointer2D;
}

void jfa_5(unsigned int B, unsigned int T, IdVec2 *pointers, int w) {
    int s = w/2;
    while (0 < s) {
        jfa_impl_5<1, 0><<<B, T>>>(pointers, w, s);
        s /= 2;
    }
    s = w/2;
    while (0 < s) {
        jfa_impl_5<0, 1><<<B, T>>>(pointers, w, s);
        s /= 2;
    }
}

void jfa_6(unsigned int B, unsigned int T, int *pointers, int *pointers2, int w) {
    assert(w % T == 0);
    constexpr const int Th = 512;

    for (int i = 0; i < 2; ++i) {
        int s = w/2;
        while (Th < s) {
            jfa_impl_3<1, 0><<<B, T>>>(pointers, w, s);
            s /= 2;
        }
        jfa_impl_4_horizontal<Th><<<w*w / Th, Th>>>(pointers, w);

        // auto dimBlock = dim3(32, 4);
        // auto dimGrid = dim3(w / 32, w / 32);
        // transpose_pointers<<<dimGrid, dimBlock>>>(pointers2, pointers, w);
        uvec2 t {4, 32};
        transpose_pointers_naive<<<{w/t.x, w/t.y, 1u}, {t.x, t.y, 1u}>>>(pointers2, pointers, w);
        std::swap(pointers, pointers2);
    }
}
