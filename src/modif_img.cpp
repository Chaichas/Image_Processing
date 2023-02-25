#include <stdio.h>
#include <string.h>
#include <algorithm>
#include "FreeImage.h"
#include "../inc/mykernel_pixel_saturation.h"
#include "../inc/mykernel_horizontal_symmetry.h"
#include "../inc/mykernel_blur_image.h"
#include "../inc/mykernel_grayscale.h"
#include "../inc/mykernel_sobel.h"

// Define the block width of the grid (unsigned)
#define BLOCK_WIDTH 32

#define WIDTH 1920
#define HEIGHT 1024*
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
  
  /* Question 6 */
  //run_pixel_saturation(d_img, width, height, BLOCK_WIDTH);
  /* END Question 6 */

  /* Question 7 */
  //run_horizontal_symmetry(d_img, d_tmp, width, height, BLOCK_WIDTH);
  /* END Question 7 */

  /* Question 8 */
  //run_blur_image(d_img, width, height, BLOCK_WIDTH);
  /* END Question 8 */

  /* Question 9 */
  //run_grayscale_image(d_img, width, height, BLOCK_WIDTH);
  /* END Question 9 */

  /* Question 10 */
  //run_sobel_filter(d_img, width, height, BLOCK_WIDTH);
  /* END Question 10 */

  /* Question 11 - a */

  // Initialize the resized output matrix
  unsigned width_out = width / 3;
  unsigned height_out = height / 3;
  unsigned int* d_img_out = new unsigned int[3 * width_out * height_out]; // allocate array "d_img_out"
  std::fill(d_img_out, d_img_out + (3 * width_out * height_out), 0);  // Fill "d_img_out" with zeros

  run_resize_image(d_img, d_img_out, width, height, width_out, height_out, BLOCK_WIDTH);

  /* END Question 11 - a */


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
