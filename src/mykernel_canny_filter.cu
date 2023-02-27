/**/
#include "../inc/mykernel_canny_filter.h"

__global__ void canny_filter(unsigned char *img, unsigned char *tmp, int width, int height)
{
    // Calculate the thread indices within a 2D grid
    int idx_col = threadIdx.x + blockDim.x * blockIdx.x;
    int idx_row = threadIdx.y + blockDim.y * blockIdx.y;

    // Calculate the thread index
    int idx = ((idx_line * width) + idx_col) * 3;

    // Calculate the gradient magnitude
    float mag_gradient = (float)sqrtf(powf((float)tmp[idx], 2) 
                                + powf((float)tmp[idx + 1], 2) 
                                + powf((float)tmp[idx + 2], 2));

    // Double thresholding (low and high) => luminosity area
    if (mag_gradient < LOW_THRESHOLD) {
        img_out[idx + 0] = 0;
        img_out[idx + 1] = 0;
        img_out[idx + 2] = 0;
    }
    else if (mag_gradient > HIGH_THRESHOLD) {
        img_out[idx + 0] = 255;
        img_out[idx + 1] = 255;
        img_out[idx + 2] = 255;
    }
    else {
        img_out[idx + 0] = 128;
        img_out[idx + 1] = 128;
        img_out[idx + 2] = 128;
    }
}

