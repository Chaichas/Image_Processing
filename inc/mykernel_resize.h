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

// Define scaling factor
# define SCALE_FACTOR (0.5)

/* Question 11 -a */

void run_resize_image(unsigned int *d_img_in, unsigned int *d_img_out, unsigned width_init, unsigned height_init, unsigned width_out, unsigned height_out, unsigned BLOCK_WIDTH);

/* END Question 11 -a */

