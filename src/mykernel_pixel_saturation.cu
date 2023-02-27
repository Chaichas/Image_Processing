/* Question 6 */

#include "../inc/mykernel_pixel_saturation.h"

/*  Pixels saturation kernel on GPU */
__global__ void pixel_saturation(unsigned int *img, unsigned width, unsigned height, unsigned saturation){
  
  // Calculate the thread indices within a 2D grid
  int idx_col = threadIdx.x + blockIdx.x * blockDim.x;
  int idx_line = threadIdx.y + blockIdx.y * blockDim.y;

  // Calculate the thread index
  int idx = ((idx_line * width) + idx_col) * 3;

  if ((idx_col < width) && (idx_line < height)){

    unsigned int pixel = 255; //max

    // Saturation of the red pixel
    if (saturation == 0){
      img[idx + 0] = max(img[idx + 0], pixel);
    }
    // Saturation of the blue pixel
    if (saturation == 1){
      img[idx + 1] = max(img[idx + 1], pixel);
    }
    // Saturation of the green pixel
    if (saturation == 2){
      img[idx + 2] = max(img[idx + 2], pixel);
    }
  }
}

/*  Run of the pixels saturation kernel */
void run_pixel_saturation(unsigned int *d_img, unsigned width, unsigned height, unsigned BLOCK_WIDTH) {
    
  // CUDA events to measure the execution time of the popArt kernel
  /*cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start); */

  // Memory allocation on device (GPU)
  unsigned int *dk_img;
  //cudaMalloc((void **)&dk_img, );
  CUDA_VERIF(cudaMalloc((void **)&dk_img, sizeof(unsigned int) * 3 * width * height));
  
  // Transfer data from CPU to GPU
  //cudaMemcpy(dk_img, d_img, sizeof(unsigned int) * 3 * width * height, cudaMemcpyHostToDevice);
  CUDA_VERIF(cudaMemcpy(dk_img, d_img, sizeof(unsigned int) * 3 * width * height, cudaMemcpyHostToDevice));
  
  /*
    - Define the x-dimension of the grid
    - Take into account if the  "width" is not divided by "BLOCK_WIDTH"
  */
  int nb_block_x = width / BLOCK_WIDTH;
  if(width % BLOCK_WIDTH) nb_block_x++;

  /*
    - Define the y-dimension of the grid
    - Take into account if the  "height" is not divided by "BLOCK_WIDTH"
  */
  int nb_block_y = height / BLOCK_WIDTH;
  if(height % BLOCK_WIDTH) nb_block_y++;

  /*
    - Define the 2D grid size using dim3 structure : number of blocks
    - Define the size of each block using dim3 structure : number of threads in the block
  */
  dim3 grid_size(nb_block_x, nb_block_y);
  dim3 block_size(BLOCK_WIDTH, BLOCK_WIDTH);

  /*
    Define the saturation channel: R (0), G (1) or B (2)
  */
  int saturation = 0;

  // Calling "kernel_saturate_pixel" 
  pixel_saturation<<<grid_size, block_size>>>(dk_img, width, height, saturation);
  CUDA_VERIF(cudaDeviceSynchronize()); //synchronization

  // Transfer data back from GPU to CPU
  //cudaMemcpy(d_img, dk_img, sizeof(unsigned int) * 3 * width * height, cudaMemcpyDeviceToHost);
  CUDA_VERIF(cudaMemcpy(d_img, dk_img, sizeof(unsigned int) * 3 * width * height, cudaMemcpyDeviceToHost));

  /*cudaEventRecord(stop); 
  cudaEventSynchronize(stop);
  float elapsed_ms = 0;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  printf("Image of size: %dx%d\n\tExecuted with time: %f s\n", width, height, elapsed_ms/1000); */

  // Free allocated memory on GPU
  cudaFree(dk_img);

}

/* END Question 6 */