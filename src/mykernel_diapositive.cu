/* Question 11 - c : Diapositive effect */

#include "../inc/mykernel_diapositive.h"

/*  Diapositive effect kernel on GPU */
__global__ void diapositive_effect(unsigned int *img, unsigned int width, unsigned int height){
    
    // Calculate the thread indices within a 2D grid
    int idx_col = threadIdx.x + blockDim.x * blockIdx.x;
    int idx_line = threadIdx.y + blockDim.y * blockIdx.y;

    // Calculate the thread index
    int idx = ((idx_line * width) + idx_col) * 3;

    // Diapositive effect
    if ((idx_col < width) && (idx_line < height)) {
        
        img[idx + 0] = 255 - img[idx + 0];
        img[idx + 1] = 255 - img[idx + 1];
        img[idx + 2] = 255 - img[idx + 2];
    }

}


/*  Run of the diapositive effect kernel */
void run_diapositive_effect(unsigned int *d_img, unsigned width, unsigned height, unsigned BLOCK_WIDTH) {
    
    // CUDA events to measure the execution time of the kernel
    /*cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start); */
    
    // Memory allocation on device (GPU)
    unsigned int *dk_img;
    CUDA_VERIF(cudaMalloc((void **)&dk_img, sizeof(unsigned int) * 3 * width * height));
  
    // Transfer data from CPU to GPU
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

    // Calling "diapositive_effect" kernel
    diapositive_effect<<<grid_size, block_size>>>(dk_img, width, height);
    CUDA_VERIF(cudaDeviceSynchronize()); //synchronization

    // Transfer data from GPU to CPU
    CUDA_VERIF(cudaMemcpy(d_img, dk_img, sizeof(unsigned int) * 3 * width * height, cudaMemcpyDeviceToHost));

    /*cudaEventRecord(stop); 
    cudaEventSynchronize(stop);
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    printf("Image of size: %dx%d\n\tExecuted with time: %f s\n", width, height, elapsed_ms/1000); */

    // Free allocated memory on GPU
    cudaFree(dk_img);

}

/* END Question 11 - c */