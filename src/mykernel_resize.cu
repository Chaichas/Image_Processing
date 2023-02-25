/* Question 11 - a */

#include "../inc/mykernel_resize.h"

/* Resize image kernel */
__global__ void resize_image(unsigned int* img_in, unsigned int* img_out, unsigned int width, unsigned int height, unsigned int width_out, unsigned int height_out) {
    
    // Calculate the thread indices within a 2D grid
    int idx_col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_line = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx_col < width_out && idx_line < height_out)
    {
        // Calculate the x and y scale
        float x_scale = (float)width / (float)width_out;
        float y_scale = (float)height / (float)height_out;
        
        // Coordinates of neighbor pixel in the original image
        int x_top = (int)(idx_col * x_scale);
        int y_top = (int)(idx_line * y_scale);
        int x_bottom = x_top + 1;
        int y_bottom = y_top + 1;

        // Coordiantes of the current pixel in the output image
        int idx_top_left = (y_top * width + x_top) * 3;
        int idx_top_right = (y_top * width + x_bottom) * 3;
        int idx_bottom_left = (y_bottom * width + x_top) * 3;
        int idx_bottom_right = (y_bottom * width + x_bottom) * 3;

        // Compute output image
        for (int i = 0; i < 3; i++)
        {
            float out_top_left = (float)img_in[idx_top_left + i];
            float out_top_right = (float)img_in[idx_top_right + i];
            float out_bottom_left = (float)img_in[idx_bottom_left + i];
            float out_bottom_right = (float)img_in[idx_bottom_right + i];

            float value = out_top_left * (1 - ((idx_col * x_scale) - x_top)) * (1 - ((idx_line * y_scale) - y_top))
                + out_top_right * ((idx_col * x_scale) - x_top) * (1 - ((idx_line * y_scale) - y_top))
                + out_bottom_left * (1 - ((idx_col * x_scale) - x_top)) * ((idx_line * y_scale) - y_top)
                + out_bottom_right * ((idx_col * x_scale) - x_top) * ((idx_line * y_scale) - y_top);

            // Output image
            img_out[(idx_line * width_out + idx_col) * 3 + i] = (unsigned int)value;
        }
    }
}

/*  Run of the resize image kernel */
void run_resize_image(unsigned int *d_img_in, unsigned int *d_img_out, unsigned width_init, unsigned height_init, unsigned width_out, unsigned height_out, unsigned BLOCK_WIDTH) {

    // Memory allocation on device (GPU)
    unsigned int *dk_img_int; //input image
    unsigned int *dk_img_out; //output image
    CUDA_VERIF(cudaMalloc((void **)&dk_img_int, sizeof(unsigned int) * 3 * width_init * height_init));
    CUDA_VERIF(cudaMalloc((void **)&dk_img_out, sizeof(unsigned int) * 3 * width_out * height_out));

    // Transfer data from GPU to CPU
    CUDA_VERIF(cudaMemcpy(dk_img_int, d_img_in, sizeof(unsigned int) * 3 * width_init * height_init, cudaMemcpyHostToDevice));
    CUDA_VERIF(cudaMemcpy(dk_img_out, d_img_out, sizeof(unsigned int) * 3 * width_out * height_out, cudaMemcpyHostToDevice));

    /*
        - Define the x-dimension of the grid
        - Take into account if the  "width_out" is not divided by "BLOCK_WIDTH"
    */
    int nb_block_x = width_out / BLOCK_WIDTH;
    if(width_out % BLOCK_WIDTH) nb_block_x++;

    /*
        - Define the y-dimension of the grid
        - Take into account if the  "height_out" is not divided by "BLOCK_WIDTH"
    */
    int nb_block_y = height_out / BLOCK_WIDTH;
    if(height_out % BLOCK_WIDTH) nb_block_y++;

    /*
        - Define the 2D grid size using dim3 structure : number of blocks
        - Define the size of each block using dim3 structure : number of threads in the block
    */
    dim3 grid_size(nb_block_x, nb_block_y);
    dim3 block_size(BLOCK_WIDTH, BLOCK_WIDTH);

    // Calling "resize_image" kernel
    resize_image<<<grid_size, block_size>>>(dk_img_int, dk_img_out, width_init, height_init, width_out, height_out);
    CUDA_VERIF(cudaDeviceSynchronize()); //synchronization

    // Transfer data from CPU to GPU
    CUDA_VERIF(cudaMemcpy(d_img_out, dk_img_out, sizeof(unsigned int) * 3 * width_out * height_out, cudaMemcpyDeviceToHost));

    // Free allocated memory on GPU
    cudaFree(dk_img_int);
    cudaFree(dk_img_out);

}

/* END Question 11 - a */

