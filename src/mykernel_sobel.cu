/* Question 10 */

#include "../inc/mykernel_sobel.h"

/************************************************************************************************/
/*  Sobel filter kernel on GPU : Using GLOBAL MEMORY! => (WORKS : uncomment to try it :) ) */
/************************************************************************************************/
/*__global__ void sobel_filter(unsigned int *img, unsigned width, unsigned height){

    // Calculate the thread indices within a 2D grid
    int idx_col = threadIdx.x + blockDim.x * blockIdx.x;
    int idx_line = threadIdx.y + blockDim.y * blockIdx.y;

    // Apply sobel filter on the image
    if ((idx_col < width) && (idx_line < height))
    {

        // Calculate the thread index
        int idx = ((idx_line * width) + idx_col) * 3;

        // Initialize gradient_x and gradient_y to null
        unsigned int  gradient_x = 0;
        unsigned int gradient_y = 0;

        // Apply the Sobel filter (gradient)
        for (int i = -1; i <= 1; i++)
        {
            for (int j = -1; j <= 1; j++)
            {

                int col = idx_col + j;
                int row = idx_line + i;

                if ((col >= 0) && (col < width) && (row >= 0) && (row < height))
                {
                    int w_idx = ((row * width) + col) * 3;
                    int weight = (i == 0 || j == 0) ? 2 : 1; // weight of the filter

                    // Calculate the gradient
                    if (i == -1)
                        gradient_y -= weight * img[w_idx];
                    else if (i == 1)
                        gradient_y += weight * img[w_idx];

                    if (j == -1)
                        gradient_x -= weight * img[w_idx];
                    else if (j == 1)
                        gradient_x += weight * img[w_idx];
                }
            }
        }
        // Calculate the magnitude of the gradient
        unsigned int  mag = sqrt((gradient_x * gradient_x) + (gradient_y * gradient_y));
        
        // Threshold the magnitude to apply sobel filter on the image
        img[idx + 0] = (mag >= THRESHOLD) ? 255 : 0;
        img[idx + 1] = (mag >= THRESHOLD) ? 255 : 0;
        img[idx + 2] = (mag >= THRESHOLD) ? 255 : 0;

    }
}*/

/************************************************************************************************/
/*  Sobel filter kernel on GPU : Using SHARED MEMORY! */
/************************************************************************************************/
__global__ void sobel_filter(unsigned int *img, unsigned width, unsigned height){
    
    // Declaration of a shared memory image
    constexpr int s_size = ((BLOCK_SIZE+ 2) * (BLOCK_SIZE + 2)) * 3;
    __shared__ unsigned int s_img[s_size];

    // Calculate the thread indices within a 2D grid
    int idx_col = threadIdx.x + blockDim.x * blockIdx.x;
    int idx_line = threadIdx.y + blockDim.y * blockIdx.y;

    // Load pixel values into shared memory
    if ((idx_col < width) && (idx_line < height))
    {
        // Calculate the shared memory indices
        int s_col = threadIdx.x + 1;
        int s_line = threadIdx.y + 1;

        // Calculate the image index
        int idx_img = ((idx_line * width) + idx_col) * 3;

        // Load the pixel value into shared memory (2 is the padding)
        s_img[(s_line * (BLOCK_SIZE + 2) + s_col) * 3] = img[idx_img + 0];
        s_img[((s_line * (BLOCK_SIZE + 2) + s_col) * 3) + 1] = img[idx_img + 1];
        s_img[((s_line * (BLOCK_SIZE + 2) + s_col) * 3) + 2] = img[idx_img + 2];

        // Load pixels into shared memory edge cases
        if (threadIdx.x == 0)
        {
            // Load left column
            if ((idx_col > 0) && (idx_line < height))
            {
                int idx_img = ((idx_line * width) + idx_col - 1) * 3;
                s_img[(s_line * (BLOCK_SIZE + 2))*3] = img[idx_img + 0];
                s_img[(s_line * (BLOCK_SIZE + 2)) * 3 + 1] = img[idx_img + 1];
                s_img[(s_line * (BLOCK_SIZE + 2)) * 3 + 2] = img[idx_img + 2];
            }
        }
        else if ((threadIdx.x == blockDim.x - 1) && (idx_col < width - 1) && (idx_line < height))
        {
            // Load right column
            int idx_img = ((idx_line * width) + idx_col + 1) * 3;
            s_img[(s_line * (BLOCK_SIZE + 2) + BLOCK_SIZE + 1) * 3] = img[idx_img + 0];
            s_img[(s_line * (BLOCK_SIZE + 2) + BLOCK_SIZE + 1) * 3 + 1] = img[idx_img + 1];
            s_img[(s_line * (BLOCK_SIZE + 2) + BLOCK_SIZE + 1) * 3 + 2] = img[idx_img + 2];
        }

        if ((threadIdx.y == 0) && (idx_line > 0) && (idx_col < width))
        {
            // Load top row
            int idx_img = ((idx_line - 1) * width + idx_col) * 3;
            s_img[(threadIdx.x + 1) * 3] = img[idx_img + 0];
            s_img[(threadIdx.x + 1) * 3 + 1] = img[idx_img + 1];
            s_img[(threadIdx.x + 1) * 3 + 2] = img[idx_img + 2];
        }

        else if (threadIdx.y == blockDim.y - 1 && idx_line < height - 1 && idx_col < width)
        {
            // Load bottom row
        int idx_img = ((idx_line + 1) * width + idx_col) *3;
        s_img[((BLOCK_SIZE + 1) * (BLOCK_SIZE + 2) + threadIdx.x + 1) * 3] = img[idx_img + 0];
        s_img[((BLOCK_SIZE + 1) * (BLOCK_SIZE + 2) + threadIdx.x + 1) * 3 + 1] = img[idx_img + 1];
        s_img[((BLOCK_SIZE + 1) * (BLOCK_SIZE + 2) + threadIdx.x + 1) * 3 + 2] = img[idx_img + 2];
        }

        // Threads synchronization
        __syncthreads();

        /* 
            Calculate the gradients on x and y
            REF : https://homepages.inf.ed.ac.uk/rbf/HIPR2/sobel.htm 
        */
        int gx = s_img[(s_line - 1) * (BLOCK_SIZE + 2) * 3 + (s_col + 1) * 3] - s_img[(s_line - 1) * (BLOCK_SIZE+2) * 3 + (s_col - 1) * 3]
            + 2 * s_img[(s_line) * (BLOCK_SIZE + 2) * 3 + (s_col + 1) * 3] - 2 * s_img[(s_line) * (BLOCK_SIZE + 2) * 3 + (s_col - 1) * 3]
            + s_img[(s_line + 1) * (BLOCK_SIZE + 2) * 3 + (s_col + 1) * 3] - s_img[(s_line + 1) * (BLOCK_SIZE + 2) * 3 + (s_col - 1) * 3];

        int gy = s_img[(s_line - 1) * (BLOCK_SIZE + 2) * 3 + (s_col - 1) * 3] - s_img[(s_line + 1) * (BLOCK_SIZE + 2) * 3 + (s_col - 1) * 3]
                + 2 * s_img[(s_line - 1)*(BLOCK_SIZE + 2) * 3 + (s_col) * 3] - 2 * s_img[(s_line + 1) * (BLOCK_SIZE + 2) * 3 + (s_col) * 3]
                + s_img[(s_line - 1) * (BLOCK_SIZE + 2) * 3 + (s_col + 1) * 3] - s_img[(s_line + 1) * (BLOCK_SIZE + 2) * 3 + (s_col + 1) * 3];

        // Define magnitude of the gradient filter
        int mag = (int)sqrtf((float)(gx*gx) + (float)(gy*gy));

        // Apply thresholding
        img[(idx_line * width * 3 + idx_col * 3) + 0] = (mag > THRESHOLD) ? 255 : 0;
        img[(idx_line * width * 3 + idx_col * 3) + 1] = (mag > THRESHOLD) ? 255 : 0;
        img[(idx_line * width * 3 + idx_col * 3) + 2] = (mag > THRESHOLD) ? 255 : 0;

    }
}



/*  Run of the sobel filter kernel */
void run_sobel_filter(unsigned int *d_img, unsigned width, unsigned height, unsigned BLOCK_WIDTH)
{

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
    if (width % BLOCK_WIDTH)
        nb_block_x++;

    /*
        - Define the y-dimension of the grid
        - Take into account if the  "height" is not divided by "BLOCK_WIDTH"
    */
    int nb_block_y = height / BLOCK_WIDTH;
    if (height % BLOCK_WIDTH)
        nb_block_y++;

    /*
        - Define the 2D grid size using dim3 structure : number of blocks
        - Define the size of each block using dim3 structure : number of threads in the block
    */
    dim3 grid_size(nb_block_x, nb_block_y);
    dim3 block_size(BLOCK_WIDTH, BLOCK_WIDTH);

    // Calling "sobel_filter" kernel
    sobel_filter<<<grid_size, block_size>>>(dk_img, width, height);
    CUDA_VERIF(cudaDeviceSynchronize()); // synchronization

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

/* END Question 10 */