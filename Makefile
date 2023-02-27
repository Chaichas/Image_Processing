# Compiler and flags                                                                                                      
GPP = g++
NVCC = nvcc
MATHS=--expt-relaxed-constexpr
IFLAGS = -I${HOME}/softs/FreeImage/include

# Compiling flags
CFLAGS = -O3 -g

# Library                                                                                                                 
LDFLAGS = -L${HOME}/softs/FreeImage/lib/ -lfreeimage

# Source files                                                                                                            
SRCDIR = src
INCDIR = inc

PROGRAM_NAME = modif_img.exe

default: all

all: $(PROGRAM_NAME)

$(PROGRAM_NAME): $(SRCDIR)/modif_img.cpp $(SRCDIR)/mykernel_pixel_saturation.cu $(SRCDIR)/mykernel_horizontal_symmetry.cu\
 $(SRCDIR)/mykernel_blur_image.cu $(SRCDIR)/mykernel_grayscale.cu $(SRCDIR)/mykernel_sobel.cu $(SRCDIR)/mykernel_resize.cu $(SRCDIR)/mykernel_rotation.cu\
 $(SRCDIR)/mykernel_popArt.cu
	$(NVCC) $(CFLAGS) $(MATHS) $(IFLAGS) $^ -o $(PROGRAM_NAME) $(LDFLAGS)

clean:
	rm -f *.o $(PROGRAM_NAME)