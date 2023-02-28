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

// Define low and high threshold
#define LOW_THRESHOLD 50
#define HIGH_THRESHOLD 200

// Define Gaussian parameters
#define SIGMA (1.0f)
#define DIAMETER_BLUR 2

/* Question 11 - e */

void run_canny_filter(unsigned int *d_img, unsigned int* d_tmp,  unsigned width, unsigned height, unsigned BLOCK_WIDTH);

/* END Question 11 - e */

