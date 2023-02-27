/* Question 9 */

#include "../inc/mykernel_grayscale.h"

/*  Grayscale image kernel on GPU */
__global__ void grayscale_image(unsigned int *img, unsigned int width, unsigned int height){
    
    // Calculate the thread indices within a 2D grid
    int idx_col = threadIdx.x + blockDim.x * blockIdx.x;
    int idx_line = threadIdx.y + blockDim.y * blockIdx.y;

    // Calculate the thread index
    int idx = ((idx_line * width) + idx_col) * 3;

    // Calculate the grayscale index (as a weighted sum)
    int idx_gray = 0.299 * img[idx + 0] + 0.587 * img[idx + 1]+ 0.114 * img[idx + 2];

    // Grayscale transformation of the image
    if ((idx_col < width) && (idx_line < height)) {
        
        img[idx + 0] = idx_gray;
        img[idx + 1] = idx_gray;
        img[idx + 2] = idx_gray;
    }

}


/*  Run of the Grayscale image kernel */
void run_grayscale_image(unsigned int *d_img, unsigned width, unsigned height, unsigned BLOCK_WIDTH) {
    
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

    // Calling "grayscale_image" kernel
    grayscale_image<<<grid_size, block_size>>>(dk_img, width, height);
    CUDA_VERIF(cudaDeviceSynchronize()); //synchronization

    // Transfer data from GPU to CPU
    CUDA_VERIF(cudaMemcpy(d_img, dk_img, sizeof(unsigned int) * 3 * width * height, cudaMemcpyDeviceToHost));

    // Free allocated memory on GPU
    cudaFree(dk_img);

}

/* END Question 9 */