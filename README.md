# Canny-edge-detector-with-CUDA

# Requirements
NVIDIA GPU with CUDA support  
CUDA Toolkit installed  
Python 3.x  
Conda environment with Numba and Pillow packages  

# Usage
1. Activate the Conda environment with Numba: conda activate numba  
2. Run the main Python script: python main.py [--tb int] [--bw] [--gauss] [--sobel] [--threshold]  <inputImage> <outputImage>  
  
inputImage :  the source image  
outputImage : the destination image    
--tb int : optional size of a thread block for all operations  
--bw : perform only the bw_kernel  
--gauss : perform the bw_kernel and the gauss_kernel   
--sobel : perform all kernels up to sobel_kernel  and write to disk the magnitude of each pixel  
--threshold : perform all kernels up to threshold_kernel  
