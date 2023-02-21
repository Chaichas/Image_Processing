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
      std::cerr << "Sorry, CUDA runtime failed. For more details: Error %s, line %d \n, file %s" << (cudaGetErrorString(cudaStatus), __LINE__, __FILE__) << std::endl; \
      exit(1); \
  } \
}

/* Question 6 */

void run_pixel_saturation(unsigned int *d_img, unsigned width, unsigned height, unsigned BLOCK_WIDTH);

/* END Question 6 */
