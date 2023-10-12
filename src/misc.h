#pragma once

#include <glm/glm.hpp>
#include <functional>
#include <chrono>

__device__
inline glm::vec3 rgb2hsv(glm::vec3 c) {
    using namespace glm;
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(
        vec4(vec2(c.b, c.g), vec2(K.w, K.z)),
        vec4(vec2(c.g, c.b), vec2(K.x, K.y)),
        step(c.b, c.g)
    );
    vec4 q = mix(vec4(vec3(p.x, p.y, p.w), c.r), vec4(c.r, vec3(p.y, p.z, p.x)), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0f * d + e)), d / (q.x + e), q.x);
}

__device__
inline glm::vec3 hsv2rgb(glm::vec3 c) {
    using namespace glm;
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(vec3(c.x) + vec3(K)) * 6.0f - vec3(K.w));
    return c.z * mix(vec3(K.x), clamp(p - vec3(K.x), vec3(0.0), vec3(1.0)), c.y);
}

__device__
inline float length_squared(glm::vec2 v) {
    return v.x*v.x + v.y*v.y;
}

inline float perf(std::function<void()> fn) {
    using namespace std::chrono;

    cudaDeviceSynchronize();
    auto start = steady_clock::now();
    fn();
    cudaDeviceSynchronize();
    auto end = steady_clock::now();

    return duration_cast<microseconds>(end - start).count() / 1000.0f;
}

#define CHECK_LAST_CUDA_ERROR() checkLastError(__FILE__, __LINE__)
inline void checkLastError(const char* const file, const int line) {
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl
            << cudaGetErrorString(err) << std::endl;
    }
}
