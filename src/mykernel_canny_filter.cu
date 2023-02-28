/* 
    Question 11-e 
    REF : https://fr.wikipedia.org/wiki/Filtre_de_Canny
*/

#include "../inc/mykernel_canny_filter.h"

/* Gaussian for noise reduction (image smoothing) */
__device__ float gaussian_operator(int x, int y) {
    return expf(-(x * x + y * y) / (2 * SIGMA * SIGMA)); // weight of the pixel
}

/* The Canny filter kernel (divided to steps) */
__global__ void canny_filter(unsigned int* img, unsigned int* tmp, unsigned height, unsigned width) {
    
    
    // Calculate the thread indices within a 2D grid
    int idx_col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if thread indices are within image bounds
    if ((idx_col < width) && (idx_row < height)) {
    

        // Calculate thread index
        int idx = ((idx_row * width) + idx_col) * 3;

        /*******************************************************************************************************************************/    
        // STEP 1 : Gaussian BLUR for image smoothing => Remove small details non-needed for edge detection
        /*******************************************************************************************************************************/
        float pixel_red = (0.0f), pixel_green = (0.0f), pixel_blue = (0.0f);
        float sum_weight = (0.0f);

        // Apply the Gaussian filter
        for (int y = -DIAMETER_BLUR; y <= DIAMETER_BLUR; y++) {
            for (int x = -DIAMETER_BLUR; x <= DIAMETER_BLUR; x++) {
                
                // Calculate indices of the neighboring pixels
                int x_neighbor = idx_col + x;
                int y_neighbor = idx_row + y;

                if (((x_neighbor >= 0) && (x_neighbor < width)) && ((y_neighbor >= 0) && (y_neighbor < height))) {
                    int idx_neighbor = ((y_neighbor * width) + x_neighbor) * 3;
                    float weight = gaussian_operator(x, y);  // Gaussian function
                    sum_weight += weight; // Accumulated sum of weights

                    // Weighted sum for neighboring pixels for each color channel
                    pixel_red += tmp[idx_neighbor + 0] * weight;
                    pixel_green += tmp[idx_neighbor + 1] * weight;
                    pixel_blue += tmp[idx_neighbor + 2] * weight;
                }
            }
        }

        // Output image gets smoothed pixel values 
        img[idx + 0] = (unsigned int)(pixel_red / sum_weight);
        img[idx + 1] = (unsigned int)(pixel_green / sum_weight);
        img[idx + 2] = (unsigned int)(pixel_blue / sum_weight);

        /*******************************************************************************************************************************/        
        // STEP 2 : Apply Sobel filter to calculate the vertical and horizontal changes in the image intensity
        /*******************************************************************************************************************************/

        /************************************************** - Gradient Itensity - **************************************************/
        // Calculate neighbors indices (within image limits)
        int idx_right = ((idx_row * width) + min(idx_col + 1, width - 1)) * 3;
        int idx_left = ((idx_row * width) + max(idx_col - 1, 0)) * 3;
        int idx_bottom_left = (max(idx_row - 1, 0) * width + max(idx_col - 1, 0)) * 3;
        int idx_bottom_right = (max(idx_row - 1, 0) * width + min(idx_col + 1, width - 1)) * 3;
        int idx_top_left = (min(idx_row + 1, height - 1) * width + max(idx_col - 1, 0)) * 3;
        int idx_top_right = (min(idx_row + 1, height - 1) * width + min(idx_col + 1, width - 1)) * 3;
        int idx_top = (min(idx_row + 1, height - 1) * width + idx_col) * 3;
        int idx_bottom = (max(idx_row - 1, 0) * width + idx_col) * 3;

        // Calculate the gradients (for the 3 channels)
        float x_gradient_r = -tmp[idx_bottom_left] - 2 * tmp[idx_left] - tmp[idx_top_left]
            + tmp[idx_bottom_right] + 2 * tmp[idx_right] + tmp[idx_top_right];

        float y_gradient_r = -tmp[idx_bottom_left] - 2 * tmp[idx_bottom] - tmp[idx_bottom_right]
            + tmp[idx_top_left] + 2 * tmp[idx_top] + tmp[idx_top_right];

        float x_gradient_g = -tmp[idx_bottom_left + 1] - 2 * tmp[idx_left + 1] - tmp[idx_top_left + 1]
            + tmp[idx_bottom_right + 1] + 2 * tmp[idx_right + 1] + tmp[idx_top_right + 1];

        float y_gradient_g = -tmp[idx_bottom_left + 1] - 2 * tmp[idx_bottom + 1] - tmp[idx_bottom_right + 1]
            + tmp[idx_top_left + 1] + 2 * tmp[idx_top + 1] + tmp[idx_top_right + 1];

        float x_gradient_b = -tmp[idx_bottom_left + 2] - 2 * tmp[idx_left + 2] - tmp[idx_top_left + 2]
            + tmp[idx_bottom_right + 2] + 2 * tmp[idx_right + 2] + tmp[idx_top_right + 2];

        float y_gradient_b = -tmp[idx_bottom_left + 2] - 2 * tmp[idx_bottom + 2] - tmp[idx_bottom_right + 2]
            + tmp[idx_top_left + 2] + 2 * tmp[idx_top  + 2] + tmp[idx_top_right + 2];

        // Compute gradient magnitude (for the 3 channels) 
        float magnitude_r = sqrtf(x_gradient_r * x_gradient_r + y_gradient_r * y_gradient_r);
        float magnitude_g = sqrtf(x_gradient_g * x_gradient_g + y_gradient_g * y_gradient_g);
        float magnitude_b = sqrtf(x_gradient_b * x_gradient_b + y_gradient_b * y_gradient_b);

        // Calculate the edge_orientation (for the 3 channels)
        float edge_orientation_r = atan2f(y_gradient_r, x_gradient_r);
        float edge_orientation_g = atan2f(y_gradient_g, x_gradient_g);
        float edge_orientation_b = atan2f(y_gradient_b, x_gradient_b);


        // Write gradient magnitude and edge_orientation to the image
        img[idx + 0] = (unsigned int)(magnitude_r); //red
        img[idx + 1] = (unsigned int)(magnitude_g); //green
        img[idx + 2] = (unsigned int)(magnitude_b); //blue


        /************************************************** - Edges Orientation - **************************************************/
        // Compute the orientation angle for the pixel strongest edge
        float max_edge_orientation = (0.0f);
        if ((fabsf(edge_orientation_r) > fabsf(edge_orientation_g)) && (fabsf(edge_orientation_r) > fabsf(edge_orientation_b))) {
            max_edge_orientation = edge_orientation_r; //red
        }
        else if (fabsf(edge_orientation_g) > fabsf(edge_orientation_b)) {
            max_edge_orientation = edge_orientation_g; //green
        }
        else {
            max_edge_orientation = edge_orientation_b; //blue
        }

        // Angle equal to one of the 4 possible directions
        int orientation = 0;
        float angle4 = PI/4;
        if ((max_edge_orientation >= -angle4) && (max_edge_orientation < angle4)) {
            orientation = 0; // Horizontal edge
        }
        else if ((max_edge_orientation >= angle4) && (max_edge_orientation < (3*angle4))){
            orientation = 1; // Diagonal edge
        }
        else if ((max_edge_orientation >= (-3*angle4)) && (max_edge_orientation < -angle4)) {
            orientation = 2; // Anti-diagonal edge
        }
        else {
            orientation = 3; // Vertical edge
        }

        // Set the orientation channel of "img"
        img[idx + 3 * orientation + 0] = (255u);
        img[idx + 3 * orientation + 1] = (255u);
        img[idx + 3 * orientation + 2] = (255u);

        /*******************************************************************************************************************************/
        //step 3 : Non maxima suppression
        /*******************************************************************************************************************************/

        // Compute neighboring pixel indices based on gradient direction
        int neig1_x, neig1_y; // Neighboring pixels 1 along edge direction
        int neig2_x, neig2_y; // Neighboring pixels 2 along edge direction
        
        // Define angles orthogonal to the gradient direction
        float angle1 = -3 * PI / 8 ; //top quadrant of the unit circle
        float angle2 = 5 * PI / 8; // Bottom quadrant of the unit circle
        float angle3 = - PI / 8; // Right + left quadrant of the unit circle
        float edge_orientation;
        
        if ((max_edge_orientation == edge_orientation_r) || (max_edge_orientation == edge_orientation_g) || (max_edge_orientation == edge_orientation_b)) {
            edge_orientation = max_edge_orientation;
        }

        if ((edge_orientation < angle1) || (edge_orientation >= angle2)) {
            neig1_x = neig2_x = idx_col;
            neig1_y = idx_row - 1;
            neig2_y = idx_row + 1;
        }
        else if ((edge_orientation >= angle1) && (edge_orientation < angle3)) {
            neig1_x = (idx_col > 0) ? idx_col - 1 : 0;
            neig1_y = (idx_row > 0) ? idx_row - 1 : 0;
            neig2_x = (idx_col < width - 1) ? idx_col + 1 : width - 1;
            neig2_y = (idx_row < height - 1) ? idx_row + 1 : height - 1;
        }
        else if ((edge_orientation >= angle3) && (edge_orientation < -angle3)) {
            neig1_x = (idx_col > 0) ? idx_col - 1 : 0;
            neig2_x = (idx_col < width - 1) ? idx_col + 1 : width - 1;
            neig1_y = neig2_y = idx_row;
        }
        else {
            neig1_x = (idx_col > 0) ? idx_col - 1 : 0;
            neig1_y = (idx_row < height - 1) ? idx_row + 1 : height - 1;
            neig2_x = (idx_col < width - 1) ? idx_col + 1 : width - 1;
            neig2_y = (idx_row > 0) ? idx_row - 1 : 0;
        }

        // Get intensity values at neighboring pixels for all 3 color channels
        int idx_n1 = (neig1_y * width + neig1_x) * 3; //index neighbor 1
        unsigned int red_intensity1 = tmp[idx_n1 + 0];
        unsigned int green_intensity1 = tmp[idx_n1 + 1];
        unsigned int blue_intensity1 = tmp[idx_n1 + 2];

        int idx_n2 = (neig2_y * width + neig2_x) * 3; //index neighbor 2
        unsigned int red_intensity2 = tmp[idx_n2 + 0];
        unsigned int green_intensity2 = tmp[idx_n2 + 1];
        unsigned int blue_intensity2 = tmp[idx_n2 + 2];

        unsigned int red_intensity = tmp[idx + 0];
        unsigned int green_intensity = tmp[idx+ 1];
        unsigned int blue_intensity = tmp[idx + 2];

        // Calculate the maximum intensity value across the color channels
        unsigned int max_intensity1 = fmaxf(red_intensity1, fmaxf(green_intensity1, blue_intensity1));
        unsigned int max_intensity2 = fmaxf(red_intensity2, fmaxf(green_intensity2, blue_intensity2));

        // Check if intensity is a local maximum in the edge_orientation of the gradient
        if ((red_intensity > max_intensity1) && (red_intensity > max_intensity2)) {
            img[idx + 0] = (red_intensity);
            img[idx + 1] = (green_intensity);
            img[idx + 2] = (blue_intensity);
        }
        else {
            img[idx + 0] = (0u);
            img[idx + 1] = (0u);
            img[idx + 2] = (0u);
        }

        /*******************************************************************************************************************************/
        //step 4 : Hysteresis Double-Thresholding
        /*******************************************************************************************************************************/
        
        // Calculate the gradient intensity
        float gradient_intensity = sqrtf(img[idx + 0] * img[idx + 0] + img[idx + 1] * img[idx + 1] + img[idx + 2] * img[idx + 2]);

        if (gradient_intensity > HIGH_THRESHOLD) {
            // Considered as strong edges
            img[idx + 0] = (255u);
            img[idx + 1] = (255u);
            img[idx + 2] = (255u);
        }
        else if (gradient_intensity < LOW_THRESHOLD) {
            // Considered as non - edges
            img[idx + 0] = (0u);
            img[idx + 1] = (0u);
            img[idx + 2] = (0u);
        }
        else {
            // Edge tracking : weak edges that are connected to strong edges
            bool is_strong_edge = false;
            for (int j = -1; j <= 1; j++) {
                for (int i = -1; i <= 1; i++) {
                    
                    int neighbor_x = idx_col + i;
                    int neighbor_y = idx_row + j;
                    
                    if (((neighbor_x >= 0) && (neighbor_x < width)) && ((neighbor_y >= 0) && (neighbor_y < height))) {
                        
                        int idx_neighbor = ((neighbor_y * width) + neighbor_x) * 3;
                        float neighbor_intensity = sqrtf(img[idx_neighbor + 0] * img[idx_neighbor + 0] +
                                                        img[idx_neighbor + 1] * img[idx_neighbor + 1] +
                                                        img[idx_neighbor + 2] * img[idx_neighbor + 2]);

                        if (neighbor_intensity > HIGH_THRESHOLD) {
                            is_strong_edge = true;
                            break;
                        }
                    }
                }
                if (is_strong_edge) {
                    break;
                }
            }

            if (is_strong_edge) {
                // Strong edge
                img[idx + 0] = (255u);
                img[idx + 1] = (255u);
                img[idx + 2] = (255u);
            }
            else {
                // Non - edge
                img[idx + 0] = (0u);
                img[idx + 1] = (0u);
                img[idx + 2] = (0u);
            }
        }


    }

}


/*  Run of the Canny filter kernel */
void run_canny_filter(unsigned int *d_img, unsigned int* d_tmp,  unsigned width, unsigned height, unsigned BLOCK_WIDTH) {

    // CUDA events to measure the execution time of the kernel
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); //start

    // Memory allocation on GPU
    unsigned int *dk_img;
    unsigned int *dk_tmp;
    CUDA_VERIF(cudaMalloc((void **)&dk_img, sizeof(unsigned int) * 3 * width * height));
    CUDA_VERIF(cudaMalloc((void **)&dk_tmp, sizeof(unsigned int) * 3 * width * height));

    // Transfer data from CPU to GPU
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

    // Calling "canny_filter" kernel
    canny_filter<<<grid_size, block_size>>>(dk_img, dk_tmp, height, width);
    
    // Transfer data back from GPU to CPU
    CUDA_VERIF(cudaMemcpy(d_img, dk_img, sizeof(unsigned int) * 3 * width * height, cudaMemcpyDeviceToHost));

    cudaEventRecord(stop); //stop

    // Measure elapsed time
    cudaEventSynchronize(stop);
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    printf("Image of size: %dx%d\n\tExecuted with time: %f s\n", width, height, elapsed_ms/1000); //Execution time

    // Free allocated memory on GPU
    cudaFree(dk_img);
    cudaFree(dk_tmp);

}

/* END Question 11-e */