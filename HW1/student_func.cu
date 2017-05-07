#include "utils.h"

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numPixels)
{
  size_t pixelLocation = blockIdx.x*1024+threadIdx.x;
  if(pixelLocation<numPixels)
    greyImage[pixelLocation] = .299f * rgbaImage[pixelLocation].x + .587f * rgbaImage[pixelLocation].y + .114f * rgbaImage[pixelLocation].z;
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
  int numPixels=numRows*numCols;
  const dim3 blockSize(1024,1 , 1);
  const dim3 gridSize( numPixels/1024+ numPixels%1024, 1, 1);
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numPixels);  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}
