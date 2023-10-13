#include <iostream>
#include <vector>
#include <memory>
#include <chrono>

#include "jfa.h"
#include "display.h"
#include "misc.h"

// constexpr const unsigned int W = 512;
constexpr const unsigned int W = 1<<10;
constexpr const unsigned int S = W*W;

constexpr const unsigned int T = std::min(128u, W);
constexpr const unsigned int B = S/T;
static_assert(T * B == S);

// A single iteration of Bob Jenkins' One-At-A-Time hashing algorithm.
__device__
unsigned int hash(unsigned int x) {
    x += ( x << 10u );
    x ^= ( x >>  6u );
    x += ( x <<  3u );
    x ^= ( x >> 11u );
    x += ( x << 15u );
    return x;
}

__global__
void create_input(bool *input, float seed) {
    unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    auto y = gid / W;
    input[gid] = hash(gid + y*0.1f) % 50000 == 0;
}

__global__
void visualize(
    glm::vec4 *output,
    [[maybe_unused]] bool *input,
    [[maybe_unused]] int *pointers,
    [[maybe_unused]] float *sdf,
    glm::ivec2 viewport
) {
    unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    auto uv = glm::ivec2(gid % viewport.x, gid / viewport.x) % glm::ivec2(W);
    auto i = uv.x + uv.y * W;
    [[maybe_unused]] auto l = sdf[i];
    [[maybe_unused]] auto b = input[i];
    [[maybe_unused]] auto p = pointers[i] / static_cast<float>(S);

    if (gid < viewport.x*viewport.y) {
        float h = (hash(pointers[i]      ) % S) / static_cast<float>(S);
        float s = (hash(pointers[i] + S  ) % S) / static_cast<float>(S) * 0.8f + 0.2f;
        float v = (hash(pointers[i] + S*2) % S) / static_cast<float>(S) * 0.8f + 0.2f;
        output[gid] = glm::vec4 (
            hsv2rgb(glm::vec3(h, !b && uv.x % 128 != 0 && uv.y % 128 != 0, 0.3f + b * 0.7f)),
            // 1.0f - l * 100.0f,
            // b,
            // p,
            1.f
        );
    }
}

int main()
{
    std::cout << "width: " << W << std::endl;
    std::cout << "pixels: " << S << std::endl;

    // device resources
    bool *input;
    int *pointers;
    int *pointers2;
    float *sdf;
    glm::vec4 *output;

    // host resources
    auto output_h = std::vector<glm::vec4>();

    auto resources_created = false;
    auto create_resources = [&] (glm::ivec2 viewport) {
        assert(!resources_created);

        cudaMalloc(&input, S * sizeof(bool));
        cudaMalloc(&pointers, S * sizeof(int));
        cudaMalloc(&pointers2, S * sizeof(int));
        cudaMalloc(&sdf, S * sizeof(float));
        cudaMalloc(&output, viewport.x * viewport.y * sizeof(glm::vec4));
        output_h.resize(viewport.x * viewport.y);
        cudaDeviceSynchronize();

        resources_created = true;
    };

    auto destroy_resources = [&] {
        if (resources_created) {
            cudaDeviceSynchronize();
            cudaFree(input);
            cudaFree(pointers);
            cudaFree(pointers2);
            cudaFree(sdf);
            cudaFree(output);
        }
        resources_created = false;
    };

    auto update_input = [&] ([[maybe_unused]] float elapsed_time) {
        create_input<<<B, T>>>(input, elapsed_time);
    };

    auto perf_elapsed = 0.0f;
    auto perf_count = 0.0f;
    auto update_sdf = [&] () {
        jfa_init_pointers<<<B, T>>>(pointers, input, W);
        // jfa_init_pointers_2D<<<B, T>>>((IdVec2*)pointers, input, W);
        perf_elapsed += perf( [&] {
            // jfa_2(B, T, pointers, W);
            jfa_6(B, T, pointers, pointers2, W);
            // jfa_5(B, T, (IdVec2*)pointers, W);
        } );
        ++perf_count;
        jfa_to_sdf<<<B, T>>>(sdf, pointers, W);
    };

    auto update_output = [&] (glm::ivec2 viewport) {
        // TODO: use cuda opengl interop to reduce copies
        auto s = viewport.x * viewport.y;
        int t = 128;
        int b = (s-1)/t+1;
        visualize<<<b, t>>>(output, input, pointers, sdf, viewport);
        cudaMemcpy(output_h.data(), output, s * sizeof(glm::vec4), cudaMemcpyDeviceToHost);
    };

    using namespace std::chrono;
    auto start_time = steady_clock::now();

    auto prev_viewport = glm::ivec2 {-1, -1};
    auto update = [&] (glm::ivec2 viewport) -> glm::vec4* {
        if (viewport != prev_viewport) {
            destroy_resources();
            create_resources(viewport);

            prev_viewport = viewport;
        }

            auto now = steady_clock::now();
            auto elapsed_time = duration_cast<milliseconds>(now - start_time).count() / 1000.f;

            update_input(elapsed_time);
            update_sdf();
            update_output(viewport);

            CHECK_LAST_CUDA_ERROR();

        return output_h.data();
    };
    display(update);

    destroy_resources();

    std::cout << (perf_elapsed / perf_count) << std::endl;

    return 0;
}
