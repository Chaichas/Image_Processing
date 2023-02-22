# Compiler flags
GPP = g++
NVCC = nvcc
IFLAGS = -I${HOME}/softs/FreeImage/include

# Library
LDFLAGS = -L${HOME}/softs/FreeImage/lib/ -lfreeimage

# Source files
SRCDIR = src
INCDIR = inc

PROGRAM_NAME = modif_img.exe

default: all

all: $(PROGRAM_NAME)

$(PROGRAM_NAME): $(SRCDIR)/modif_img.cpp $(SRCDIR)/mykernel_pixel_saturation.cu $(SRCDIR)/mykernel_horizontal_symmetry.cu $(SRCDIR)/mykernel_blur_image.cu $(SRCDIR)/mykernel_grayscale.cu
	$(NVCC) $(IFLAGS) $^ -o $(PROGRAM_NAME) $(LDFLAGS)

clean:
	rm -f *.o $(PROGRAM_NAME)
