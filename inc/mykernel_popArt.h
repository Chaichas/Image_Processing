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

/* Question 12 */

void run_kernel_popArt(unsigned int *d_img, unsigned int *d_tmp, unsigned width, unsigned height, unsigned BLOCK_WIDTH);

/* END Question 12 */