/* Question 8 */

#include "../inc/mykernel_blur_image.h"

/*  Blur image kernel on GPU */
__global__ void blur_image(unsigned int *img, unsigned width, unsigned height){
  
    // Calculate the thread indices within a 2D grid
    int idx_col = threadIdx.x + blockDim.x * blockIdx.x;
    int idx_line = threadIdx.y + blockDim.y * blockIdx.y;

    if ((idx_col < width) && (idx_line < height)){

        // Calculate the thread index
        int idx = ((idx_line * width) + idx_col) * 3;

        // Initialization
        int row, col;   
        int n_idx; 
        unsigned int sum_R = 0;
        unsigned int sum_G = 0;
        unsigned int sum_B = 0;

        // Loop on neighboring pixels
        for(int i = -BLUR_FILTER_RADIUS; i <= BLUR_FILTER_RADIUS; i++){
            for (int j = -BLUR_FILTER_RADIUS; j <= BLUR_FILTER_RADIUS; j++){

                // Compute the row and col indices of the neighboring pixel
                row = idx_line + i;
                col = idx_col + j;

                if (row < 0 || row >= height || col < 0 || col >= width)
                    continue;

                // Compute the sum of the pixels values
                n_idx = ((row * width) + col) * 3;
                sum_R += img[n_idx + 0];
                sum_G += img[n_idx + 1];
                sum_B += img[n_idx + 2];
            }
        }
        // Compute the average of the pixels values
        sum_R /= BLUR_FILTER_SIZE;
        sum_G /= BLUR_FILTER_SIZE;
        sum_B /= BLUR_FILTER_SIZE;

        // Assign the new average pixel value to the new image
        img[idx + 0] = sum_R;
        img[idx + 1] = sum_G;
        img[idx + 2] = sum_B;
    }
}


/*  Run of the blur image kernel */
void run_blur_image(unsigned int *d_img, unsigned width, unsigned height, unsigned BLOCK_WIDTH) {
    
    // CUDA events to measure the execution time of the popArt kernel
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

    // Calling "blur_image" kernel
    blur_image<<<grid_size, block_size>>>(dk_img, width, height);
    CUDA_VERIF(cudaDeviceSynchronize()); //synchronization

    // Transfer data back from GPU to CPU
    CUDA_VERIF(cudaMemcpy(d_img, dk_img, sizeof(unsigned int) * 3 * width * height, cudaMemcpyDeviceToHost));

    /*cudaEventRecord(stop); 
    cudaEventSynchronize(stop);
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    printf("Image of size: %dx%d\n\tExecuted with time: %f s\n", width, height, elapsed_ms/1000); */

    // Free allocated memory on GPU
    cudaFree(dk_img);

}

/* END Question 8 */