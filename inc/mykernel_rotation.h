 #pragma once

#include <iostream>
#include <cuda.h>
#include<cuda_runtime.h>

/*
  Check the CUDA runtime return value.
    - If cudaSuccess : OK 
    - If not : Error details and EXIT
*/
#define CUDA_VERIF(state) { \
  cudaError_t cudaStatus = state; \
  if (cudaStatus != cudaSuccess) { \
  std::cerr << "Sorry, CUDA runtime failed. For more details: Error " << cudaGetErrorString(cudaStatus) << ", line " << __LINE__ << ", file " << __FILE__ << std::endl; \
      exit(1); \
  } \
}

#define ANGLE 80
#define PI 3.14159265358979323846

/* Question 11-b */

void run_image_rotation(unsigned int *d_img, unsigned int *d_img_out, unsigned width, unsigned height, float angle_rad, unsigned BLOCK_WIDTH);

/* END Question 11-b */