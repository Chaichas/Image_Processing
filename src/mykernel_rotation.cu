/* Question 11 - b */

#include "../inc/mykernel_rotation.h"

/*  Rotated image kernel on GPU (using centered method) */
__global__ void image_rotation(unsigned int *img, unsigned int *img_out, unsigned int width, unsigned int height, float angle_rad){

    // Calculate the thread indices within a 2D grid
    int idx_col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_row = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the "cos" and "sin" for rotation matrix
    float cos_angle = cos(angle_rad);
    float sin_angle = sin(angle_rad);

    // Compute the coordinates of the center point
    int x_center = width / 2;
    int y_center = height / 2;

    // Compute indices for pixels in the new image
    int col_out =round((float) (x_center + ((idx_col - x_center) * cos_angle - (idx_row - y_center) * sin_angle)));
    int row_out = round((float)(y_center + ((idx_col - x_center) * sin_angle + (idx_row - y_center) * cos_angle)));

    /* Rotation - REF : https://homepages.inf.ed.ac.uk/rbf/HIPR2/rotate.htm  */
    // Check if corresponding pixel is within bounds of input image
    if (((col_out >= 0) && (col_out < width)) && ((row_out >= 0) && (row_out < height)))
    {
        // Compute indices of corresponding pixels in input and output images
        int idx_in = (idx_row * width + idx_col) * 3;
        int idx_out = (row_out * width + col_out) * 3;

        // Copy pixel values from input image to output image        
        img_out[idx_out + 0] = img[idx_in + 0];
        img_out[idx_out + 1] = img[idx_in + 1];
        img_out[idx_out + 2] = img[idx_in + 2];

    }
}

/*  Run of the rotated image kernel */
void run_image_rotation(unsigned int *d_img, unsigned int *d_img_out, unsigned width, unsigned height, float angle_rad, unsigned BLOCK_WIDTH) {
    
    // CUDA events to measure the execution time of the kernel
    /*cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start); */

    // Memory allocation on device (GPU)
    unsigned int *dk_img;
    unsigned int *dk_img_out;
    CUDA_VERIF(cudaMalloc((void **)&dk_img, sizeof(unsigned int) * 3 * width * height));
    CUDA_VERIF(cudaMalloc((void **)&dk_img_out, sizeof(unsigned int) * 3 * width * height));
  
    // Transfer data from CPU to GPU
    CUDA_VERIF(cudaMemcpy(dk_img, d_img, sizeof(unsigned int) * 3 * width * height, cudaMemcpyHostToDevice));
    CUDA_VERIF(cudaMemcpy(dk_img_out, d_img_out, sizeof(unsigned int) * 3 * width * height, cudaMemcpyHostToDevice));
  
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

    // Calling "image_rotation" kernel
    image_rotation<<<grid_size, block_size>>>(dk_img, dk_img_out, width, height, angle_rad);
    CUDA_VERIF(cudaDeviceSynchronize()); //synchronization

    // Transfer data from GPU to CPU
    CUDA_VERIF(cudaMemcpy(d_img_out, dk_img_out, sizeof(unsigned int) * 3 * width * height, cudaMemcpyDeviceToHost));

    /*cudaEventRecord(stop); 
    cudaEventSynchronize(stop);
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    printf("Image of size: %dx%d\n\tExecuted with time: %f s\n", width, height, elapsed_ms/1000); */
    
    // Free allocated memory on GPU
    cudaFree(dk_img);
    cudaFree(dk_img_out);

}

/* END Question 11 - b */