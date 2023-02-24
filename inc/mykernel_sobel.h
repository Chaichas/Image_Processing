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

// Define a threshold to adjust the image edge detection sensitivity
#define THRESHOLD 90
constexpr int BLOCK_SIZE = 32;

/* Question 10 */

void run_sobel_filter(unsigned int *d_img, unsigned width, unsigned height, unsigned BLOCK_WIDTH);

/* END Question 10 */

