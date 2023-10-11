#pragma once

__global__
extern void jfa_init_pointers(int *pointers, bool *input, int w);

extern void jfa_0(unsigned int B, unsigned int T, int *pointers, int w);
extern void jfa_2(unsigned int B, unsigned int T, int *pointers, int w);
extern void jfa_3(unsigned int B, unsigned int T, int *pointers, int w);
extern void jfa_4(unsigned int B, unsigned int T, int *pointers, int w);

__global__
extern void jfa_to_sdf(float *sdf, int *pointers, int w);
