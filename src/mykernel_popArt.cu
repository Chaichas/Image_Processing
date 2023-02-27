/* Question 12 */
 
#include "../inc/mykernel_popArt.h"
#include <curand_kernel.h>

/*  popArt kernel ; Inspired from the original given code */
__global__ void kernel_popArt_Warhol(unsigned int* d_img, unsigned int* d_tmp, unsigned int height,unsigned int width){
   
    int idx_col = threadIdx.x + blockDim.x * blockIdx.x;
    int idx_row = threadIdx.y + blockDim.y * blockIdx.y;

    int idx = ((idx_row * width) + idx_col) * 3;

    // Quadrant of pixels position
    bool bottom_left = (idx_col < (width/2)) && (idx_row < (height/2));
    bool bottom_right = ((idx_row < height / 2) && (width / 2 < idx_col) && (idx_col < width));
    bool top_left = ((height / 2 < idx_row) && (idx_row < height) && (idx_col < width / 2));
    bool top_right = ((height / 2 < idx_row) && (idx_row < height) && (width / 2 < idx_col) && (idx_col < width));

    // Flip the image vertically
    if (idx_col<width && idx_row < height)
    {
        int idx_inv = ((width * height) - ((idx_row * width) + idx_col)) * 3;
        d_img[idx + 0] = d_tmp[idx_inv + 0];
        d_img[idx + 1] = d_tmp[idx_inv + 1];
        d_img[idx + 2] = d_tmp[idx_inv + 2]; 
        
    }

    // Bottom left
    if (bottom_left)
    {
        d_img[idx + 0] /= 2;
        d_img[idx + 1] /= 4;
        d_img[idx + 2] = 0xFF / 1.5; //bluish teint
    }

    // Bottom right
    if (bottom_right) 
    {
        int grey = d_img[idx + 0] * 0.299 + d_img[idx + 1] * 0.587 + d_img[idx + 2] * 0.114;
        d_img[idx + 0] = grey;
        d_img[idx + 1] = grey;
        d_img[idx + 2] = grey;
    }

    // Top left
    if (top_left)
    {
        d_img[idx + 0] = 0xFF / 2; //redish teint
        d_img[idx + 1] /= 2;
        d_img[idx + 2] /= 2;       
    }

    // Top right
    if (top_right)
    {
        d_img[idx + 0] = 0xFF - d_img[idx + 0];
        d_img[idx + 1] = 0xFF / 2;
        d_img[idx + 2] /= 4;
    }
}

/*  Run of the popArt kernel */
void run_kernel_popArt(unsigned int *d_img, unsigned int* d_tmp,  unsigned width, unsigned height, unsigned BLOCK_WIDTH) {
    
    // Memory allocation
    unsigned int *dk_img;
    unsigned int *dk_tmp;

    // CUDA events to measure the execution time of the popArt kernel
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Memory allocation on device (GPU)
    CUDA_VERIF(cudaMalloc((void **)&dk_img, sizeof(unsigned int) * 3 * width * height));
    CUDA_VERIF(cudaMalloc((void **)&dk_tmp, sizeof(unsigned int) * 3 * width * height));

    // Transfer data from GPU to CPU
    CUDA_VERIF(cudaMemcpy(dk_img, d_img, sizeof(unsigned int) * 3 * width * height, cudaMemcpyHostToDevice));
    CUDA_VERIF(cudaMemcpy(dk_tmp, d_tmp, sizeof(unsigned int) * 3 * width * height, cudaMemcpyHostToDevice));
  
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

    // Calling "popArt" kernel
    cudaEventRecord(start);
    kernel_popArt_Warhol<<<grid_size, block_size>>>(dk_img, dk_tmp, height, width);
    cudaEventRecord(stop);
    
    // Transfer data from CPU to GPU
    CUDA_VERIF(cudaMemcpy(d_img, dk_img, sizeof(unsigned int) * 3 * width * height, cudaMemcpyDeviceToHost));

    cudaEventSynchronize(stop);
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    printf("Image of size: %dx%d\n\tExecuted with time: %f s\n", width, height, elapsed_ms/1000); //Execution time

    // Free allocated memory on GPU
    cudaFree(dk_img);
    cudaFree(dk_tmp);

}

/* END Question 12 */