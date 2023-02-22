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

/* 
  Define blur filer (neighboring) and tile sizes for "QUESTION 7"
  REF : https://www.nvidia.com/content/nvision2008/tech_presentations/Game_Developer_Track/NVISION08-Image_Processing_and_Video_with_CUDA.pdf
*/
#define BLUR_FILTER_RADIUS (3)
#define BLUR_FILTER_DIAMETER (BLUR_FILTER_RADIUS * 2 + 1)
#define BLUR_FILTER_SIZE (BLUR_FILTER_DIAMETER * BLUR_FILTER_DIAMETER)
#define TILE_SIZE 28

/* Question 8 */

void run_blur_image(unsigned int *d_img, unsigned width, unsigned height, unsigned BLOCK_WIDTH);

/* END Question 8 */