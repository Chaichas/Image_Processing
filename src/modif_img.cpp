#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <cmath>
#include "FreeImage.h"
#include "../inc/mykernel_pixel_saturation.h"
#include "../inc/mykernel_horizontal_symmetry.h"
#include "../inc/mykernel_blur_image.h"
#include "../inc/mykernel_grayscale.h"
#include "../inc/mykernel_sobel.h"
#include "../inc/mykernel_resize.h"
#include "../inc/mykernel_rotation.h"
#include "../inc/mykernel_popArt.h"
#include "../inc/mykernel_popArt_4streams.h"

// Define the block width of the grid (unsigned)
#define BLOCK_WIDTH 32

#define WIDTH 3840
#define HEIGHT 2160
#define BPP 24 // Since we're outputting three 8 bit RGB values => 8*3=24

using namespace std;


int main (int argc , char** argv)
{
  
  FreeImage_Initialise();
  const char *PathName = "./src/img.jpg"; // provided image
  const char *PathDest = "./src/new_img.png"; //generated image
  // load and decode a regular file
  FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(PathName);

  FIBITMAP* bitmap = FreeImage_Load(FIF_JPEG, PathName, 0);

  if(! bitmap )
    exit( 1 ); //WTF?! We can't even allocate images ? Die !

  /* Image properties */
  unsigned width  = FreeImage_GetWidth(bitmap);
  unsigned height = FreeImage_GetHeight(bitmap);
  unsigned pitch  = FreeImage_GetPitch(bitmap); 

  fprintf(stdout, "Processing Image of size %d x %d\n", width, height);

  /* Memory allocation */
  unsigned int *img = (unsigned int*) malloc(sizeof(unsigned int) * 3 * width * height);
  unsigned int *d_img = (unsigned int*) malloc(sizeof(unsigned int) * 3 * width * height);
  unsigned int *d_tmp = (unsigned int*) malloc(sizeof(unsigned int) * 3 * width * height);

  BYTE *bits = (BYTE*)FreeImage_GetBits(bitmap);
  for ( int y =0; y<height; y++)
  {
    BYTE *pixel = (BYTE*)bits;
    for ( int x =0; x<width; x++)
    {
      int idx = ((y * width) + x) * 3;
      img[idx + 0] = pixel[FI_RGBA_RED]; // red
      img[idx + 1] = pixel[FI_RGBA_GREEN]; // green
      img[idx + 2] = pixel[FI_RGBA_BLUE]; // blue
      pixel += 3;
    }
    // next line
    bits += pitch;
  }

  /* Copy data */
  memcpy(d_img, img, 3 * width * height * sizeof(unsigned int));
  memcpy(d_tmp, img, 3 * width * height * sizeof(unsigned int));

  /* Kernels (GPU) */
  
  /*****************************************************************************/
  /* Question 6 : Pixel saturation */
  //run_pixel_saturation(d_img, width, height, BLOCK_WIDTH);
  /* END Question 6 */
  /*****************************************************************************/

  /*****************************************************************************/  
  /* Question 7 : Horizontal symmetry */
  //run_horizontal_symmetry(d_img, d_tmp, width, height, BLOCK_WIDTH);
  /* END Question 7 */
  /*****************************************************************************/

  /*****************************************************************************/
  /* Question 8 : Blur image */
  //run_blur_image(d_img, width, height, BLOCK_WIDTH);
  /* END Question 8 */
  /*****************************************************************************/

  /*****************************************************************************/
  /* Question 9 : Grayscale image */
  //run_grayscale_image(d_img, width, height, BLOCK_WIDTH);
  /* END Question 9 */
  /*****************************************************************************/

  /*****************************************************************************/
  /* Question 10 : Sobel filter */
  //run_sobel_filter(d_img, width, height, BLOCK_WIDTH);
  /* END Question 10 */
  /*****************************************************************************/

  /*****************************************************************************/
  /* Question 11 - a : To run this, uncomment this block & comment the rest of the code below (from : END Question 11 - a till the end) */
  /* RESIZING of the image */

  /* Initialize the resized output matrix : Uncomment to run */
  // unsigned width_out = (unsigned)(width * SCALE_FACTOR);
  // unsigned height_out = (unsigned)(height * SCALE_FACTOR);
  // unsigned int *d_img_out = (unsigned int*) malloc(sizeof(unsigned int) * 3 * width_out * height_out);

  /* Initialize the output new image to zero : Uncomment to run */
  //memset(d_img_out, 0, sizeof(unsigned int) * 3 * width_out * height_out);

  /* Call the resize kernel : Uncomment to run */
  //run_resize_image(d_img, d_img_out, width, height, width_out, height_out, BLOCK_WIDTH);

  /* Copy back image : Uncomment to run */
  //memcpy(img, d_img_out, 3 * width_out * height_out * sizeof(unsigned int));

  /* Convert image from unsigned int to RGBQUAD : Uncomment to run */
  // RGBQUAD* pixels_out = (RGBQUAD*)malloc(sizeof(RGBQUAD) * width_out * height_out);
  // for (int y = 0; y < height_out; y++){
  //   for (int x = 0; x < width_out; x++){

  //     /* indices of int and out images */
  //     int idx_in = (y * width_out + x) * 3;
  //     int idx_out = (y * width_out + x);

  //     pixels_out[idx_out].rgbRed = img[idx_in + 0];
  //     pixels_out[idx_out].rgbGreen = img[idx_in + 1];
  //     pixels_out[idx_out].rgbBlue = img[idx_in + 2];
  //     pixels_out[idx_out].rgbReserved = 0;

  //   }
  // }

  /* Copy pixels to output bitmap : Uncomment to run */
  // FIBITMAP *bitmap_out = FreeImage_Allocate(width_out, height_out, BPP);
  // BYTE *bits_out = (BYTE*)FreeImage_GetBits(bitmap_out);
  // unsigned pitch_out = FreeImage_GetPitch(bitmap_out);
  // for (int y = 0; y < height_out; y++)
  // {
  //   BYTE* pixel = (BYTE*)bits_out;
  //   for (int x = 0; x < width_out; x++)
  //   {
  //     RGBQUAD newcolor = pixels_out[y * width_out + x];

  //     if (!FreeImage_SetPixelColor(bitmap_out, x, y, &newcolor))
  //     {
  //       fprintf(stderr, "(%d, %d) Fail...\n", x, y);
  //     }
  //     pixel += 3;
  //   }
  //   bits_out += pitch_out;
  // }

  /* Save the image as a PNG file : Uncomment to run */
  // if (FreeImage_Save(FIF_PNG, bitmap_out, PathDest, 0))
  // {
  //   cout << "Image successfully saved ! " << endl;
  // }
  // FreeImage_DeInitialise(); //Cleanup !                                                              

  /* Free allocated memory on CPU : Uncomment to run */
  // free(img);
  // free(d_img);
  // free(d_tmp);
  // free(d_img_out);

  /* END Question 11 - a */
  /*****************************************************************************/

  /*****************************************************************************/
  /* Question 11 - b : To run this, uncomment this block & comment the rest of the code below (from : END Question 11 - b till the end) */
  /* ROTATION of the image */

  /* Allocation of output image : Uncomment to run
    * Normally: width_out = (int)(abs(height * sin(angle_rad)) + abs(width * cos(angle_rad)));
    * Normally: height_out = (int)(abs(width * sin(angle_rad)) + abs(height * cos(angle_rad))); */
  //unsigned int *d_img_out = (unsigned int*) malloc(sizeof(unsigned int) * 3 * width * height);
  
  /* Initialize the output new image to zero : Uncomment to run */
  //memset(d_img_out, 0, sizeof(unsigned int) * 3 * width * height);

  // Conversion of the angle to rad : Uncomment to run
  //float angle_rad = (ANGLE * PI/ 180);
  
  // Calling of the image rotation kernel : Uncomment to run
  //run_image_rotation(d_img, d_img_out, width, height, angle_rad, BLOCK_WIDTH);
  
  /* Copy back image : Uncomment to run */
  // memcpy(img, d_img_out, 3 * width * height * sizeof(unsigned int));

  // FIBITMAP* newBitmap = FreeImage_Allocate(width, height, BPP);
  // BYTE* newBits = FreeImage_GetBits(newBitmap);
  // for (int y = 0; y < height; y++) {
  //   BYTE* pixel = (BYTE*)newBits;
  //   for (int x = 0; x < width; x++) {
  //     int idx = ((y * width) + x) * 3;
  //     pixel[FI_RGBA_RED] = img[idx + 0]; // red                                                      
  //     pixel[FI_RGBA_GREEN] = img[idx + 1]; // green                                                  
  //     pixel[FI_RGBA_BLUE] = img[idx + 2]; // blue                                                    
  //     pixel += 3;
  //   }
  //   newBits += FreeImage_GetPitch(newBitmap);
  // }

  /* Save the image as a PNG file : Uncomment to run */
  // if (FreeImage_Save(FIF_PNG, newBitmap, PathDest, 0))
  // {
  //   cout << "Image successfully saved ! " << endl;
  // }
  // FreeImage_DeInitialise(); //Cleanup !                                                           


  /* Free allocated memory on CPU : Uncomment to run */
  // free(img);
  // free(d_img);
  // free(d_tmp);
  // free(d_img_out);

  /* END Question 11 - b */
  /*****************************************************************************/

  /*****************************************************************************/
  /* Question 12 : popArt Effect */
  //run_kernel_popArt(d_img, d_tmp, width, height, BLOCK_WIDTH);
  /* END Question 12 */
  /*****************************************************************************/

  /*****************************************************************************/
  /* Question 13 : 

    The implemented code divides the image into 4 quadrants, to each modifications will be performed (saturation of the color).
    While now it is only executed on the default stream, which is stream 0, each (1/4)th of the image could be executed in a different stream,
    making it a total of 4 streams.

    Default stream (s0) : (t1) : bottom-left -> (t2) : bottom-right -> (t3) : top-left -> (t4) : top-right. 
    After : (t1) : (s0) = bottom-left, (s1) = bottom-right, (s2) : top-left, (s3) : top-right

    Using 4 streams, the modifications could be executed in parallel. Thus, we can make use of the parallel potentiel of GPU and reduce
    the amount of computation time.

    END Question 13 */
  /*****************************************************************************/

  /*****************************************************************************/
  /* Question 14 : popArt effect with 4 streams */
  run_kernel_popArt(d_img, d_tmp, width, height, BLOCK_WIDTH);
  /* END Question 14 */
  /*****************************************************************************/

  /* Copy back */
  memcpy(img, d_img, 3 * width * height * sizeof(unsigned int));

  bits = (BYTE*)FreeImage_GetBits(bitmap);
  for ( int y =0; y<height; y++)
  {
    BYTE *pixel = (BYTE*)bits;
    for ( int x =0; x<width; x++)
    {
      RGBQUAD newcolor;

      int idx = ((y * width) + x) * 3;
      newcolor.rgbRed = img[idx + 0];
      newcolor.rgbGreen = img[idx + 1];
      newcolor.rgbBlue = img[idx + 2];

      if(!FreeImage_SetPixelColor(bitmap, x, y, &newcolor))
      { fprintf(stderr, "(%d, %d) Fail...\n", x, y); }

      pixel+=3;
    }
    // next line
    bits += pitch;
  }

  /* Save the image as a PNG file */
  if( FreeImage_Save (FIF_PNG, bitmap , PathDest , 0 ))
    cout << "Image successfully saved ! " << endl ;
  FreeImage_DeInitialise(); //Cleanup !
 
  /* Free allocated memory on CPU */
  free(img);
  free(d_img);
  free(d_tmp);

  return 0;

}