/* Question 13 */

/*
    The implemented code divides the image into 4 quadrants, to each modifications will be performed (saturation of the color).
    While now it is only executed on the default stream, which is stream 0, each (1/4)th of the image could be executed in a different stream,
    making it a total of 4 streams.

    Default stream (s0) : (t1) : bottom-left -> (t2) : bottom-right -> (t3) : top-left -> (t4) : top-right. 
    After : (t1) : (s0) = bottom-left, (s1) = bottom-right, (s2) : top-left, (s3) : top-right

    Using 4 streams, the modifications could be executed in parallel. Thus, we can make use of the parallel potentiel of GPU and reduce
    the amount of computation time.

    Execution of the popArt kernl on the default stream time : 0.001153 s
    Execution of the popArt kernl on four streams time : ?

*/

/* END Question 13 */

/* Question 14 */
 
#include "../inc/mykernel_popArt_4streams.h"

/*  popArt kernel with 4 streams : Inspired from the original given code */
__global__ void kernel_popArt_Warhol_4streams(unsigned int* d_img, unsigned int* d_tmp, unsigned int height, unsigned int width, int stream_id){

    // Calculate the thread indices within a 2D grid
    int idx_col = threadIdx.x + blockDim.x * blockIdx.x;
    int idx_row = threadIdx.y + blockDim.y * blockIdx.y;

    // Calculate thread index
    int idx = ((idx_row * width) + idx_col) * 3;

    // Quadrant of pixels position (based on the "stream_id")
    bool bottom_left = (stream_id == 0) && (idx_col < (width/2)) && (idx_row < (height/2));
    bool bottom_right = (stream_id == 1) && ((idx_row < height / 2) && (width / 2 <= idx_col) && (idx_col < width));
    bool top_left = (stream_id == 2) && ((height / 2 <= idx_row) && (idx_row < height) && (idx_col < width / 2));
    bool top_right = (stream_id == 3) && ((height / 2 <= idx_row) && (idx_row < height) && (width / 2 <= idx_col) && (idx_col < width));

    if (bottom_left)
    {
        // Flip the image vertically
        int idx_inv = ((width * height) - ((idx_row * width) + idx_col)) * 3;
        d_img[idx + 0] = d_tmp[idx_inv + 0];
        d_img[idx + 1] = d_tmp[idx_inv + 1];
        d_img[idx + 2] = d_tmp[idx_inv + 2];
        
        // Saturate the blue
        d_img[idx + 0] /= 2;
        d_img[idx + 1] /= 4;
        d_img[idx + 2] = 0xFF / 1.5; //bluish teint
    }
    else if (bottom_right)
    {
        // Flip the image vertically
        int idx_inv = ((width * height) - ((idx_row * width) + idx_col)) * 3;
        d_img[idx + 0] = d_tmp[idx_inv + 0];
        d_img[idx + 1] = d_tmp[idx_inv + 1];
        d_img[idx + 2] = d_tmp[idx_inv + 2];
        
        // grayscale obtained from a weighted sum of red, green and blue
        int grey = d_img[idx + 0] * 0.299 + d_img[idx + 1] * 0.587 + d_img[idx + 2] * 0.114;
        d_img[idx + 0] = grey;
        d_img[idx + 1] = grey;
        d_img[idx + 2] = grey;
    }
    else if (top_left)
    {
        // Flip the image vertically
        int idx_inv = ((width * height) - ((idx_row * width) + idx_col)) * 3;
        d_img[idx + 0] = d_tmp[idx_inv + 0];
        d_img[idx + 1] = d_tmp[idx_inv + 1];
        d_img[idx + 2] = d_tmp[idx_inv + 2];
        
        // Saturate the red
        d_img[idx + 0] = 0xFF / 2; //redish teint
        d_img[idx + 1] /= 2;
        d_img[idx + 2] /= 2;
    }
    else if (top_right)
    {
        // Flip the image vertically
        int idx_inv = ((width * height) - ((idx_row * width) + idx_col)) * 3;
        d_img[idx + 0] = d_tmp[idx_inv + 0];
        d_img[idx + 1] = d_tmp[idx_inv + 1];
        d_img[idx + 2] = d_tmp[idx_inv + 2];
        
        //
        d_img[idx + 0] = 0xFF - d_img[idx + 0];
        d_img[idx + 1] = 0xFF / 2;
        d_img[idx + 2] /= 4;
    }
}

/*  Run of the popArt kernel with 4 streams */
void run_kernel_popArt_4streams(unsigned int *d_img, unsigned int* d_tmp, unsigned width, unsigned height, unsigned BLOCK_WIDTH) {

    // CUDA events to measure the execution time of the popArt kernel
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Declare variables
    int nb_streams = 4; // Number of streams
    unsigned int *dk_img[nb_streams];
    unsigned int *dk_tmp[nb_streams];
    cudaStream_t streams[nb_streams];
    
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

    // 
    cudaEventRecord(start); //start
    for (int i = 0; i < nb_streams; i++) {
    	
        // Create streams
        cudaStreamCreate(&streams[i]);
    	
        // Memory allocation in GPU
        CUDA_VERIF(cudaMalloc((void **)&dk_img[i], (sizeof(unsigned int) * width * height) * 3));
        CUDA_VERIF(cudaMalloc((void **)&dk_tmp[i], (sizeof(unsigned int) * width * height) * 3));

        // Transfer data from CPU to GPU (asynchronously)
        CUDA_VERIF(cudaMemcpyAsync(dk_img[i], d_img, (sizeof(unsigned int) * width * height) * 3, cudaMemcpyHostToDevice, streams[i]));        
        CUDA_VERIF(cudaMemcpyAsync(dk_tmp[i], d_tmp, (sizeof(unsigned int) * width * height) * 3, cudaMemcpyHostToDevice, streams[i]));
        
        // Calling "kernel_popArt_Warhol_4streams" kernel
        kernel_popArt_Warhol_4streams<<<grid_size, block_size, 0, streams[i]>>>(dk_img[i], dk_tmp[i], height, width, i);
        
        // Transfer data back from GPU to CPU (asynchronously)
        CUDA_VERIF(cudaMemcpyAsync(d_img, dk_img[i], (sizeof(unsigned int) * width * height) * 3, cudaMemcpyDeviceToHost, streams[i]));
    	
    	// Synchronize streams
        cudaStreamSynchronize(streams[i]);
    }
    cudaEventRecord(stop); //stop

    // Measure elapsed time
    cudaEventSynchronize(stop);
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    printf("Image of size: %dx%d\n\tExecuted with time: %f s\n", width, height, elapsed_ms/1000); //Execution time

    // Free allocated memory on GPU
    for (int i = 0; i < nb_streams; i++) {
        CUDA_VERIF(cudaStreamDestroy(streams[i])); //Destory streams
        CUDA_VERIF(cudaFree(dk_img[i]));
        CUDA_VERIF(cudaFree(dk_tmp[i]));
    }
}

/* END Question 14 */