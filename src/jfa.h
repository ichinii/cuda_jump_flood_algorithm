#pragma once

#include <glm/glm.hpp>

using IdVec2 = glm::i16vec2;

__global__
extern void jfa_init_pointers(int *pointers, bool *input, int w);
__global__
extern void jfa_init_pointers_2D(IdVec2 *pointers, bool *input, int w);

extern void jfa_0(unsigned int B, unsigned int T, int *pointers, int w);
extern void jfa_2(unsigned int B, unsigned int T, int *pointers, int w);
extern void jfa_3(unsigned int B, unsigned int T, int *pointers, int w);
extern void jfa_4(unsigned int B, unsigned int T, int *pointers, int w);
extern void jfa_5(unsigned int B, unsigned int T, IdVec2 *pointers, int w);
extern void jfa_6(unsigned int B, unsigned int T, int *pointers, int *pointers2, int w);

__global__
extern void jfa_to_sdf(float *sdf, int *pointers, int w);
