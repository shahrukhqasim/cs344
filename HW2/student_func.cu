#include "utils.h"
#include "stdio.h"

__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols, int numPixels,
                   const float* const filter, const int filterWidth)
{
  int pixelLocation = blockIdx.x*1024+threadIdx.x;
  int posX = pixelLocation%numCols;
  int posY = pixelLocation/numCols;
  if (pixelLocation >= numPixels)
    return;
  float valuesSum=0;
  short halfWidth=filterWidth/2;
  
  for (int i=0; i < filterWidth; i++) {
    for (int j=0; j < filterWidth; j++) {
		  int y = min(max(i-halfWidth+posY, 0), numRows-1);
      int x = min(max(j-halfWidth+posX, 0), numCols-1);
      int index=y*numCols+x;
      int indexFilter=i*filterWidth+j;
      valuesSum+=inputChannel[index]*filter[indexFilter];
    }
  }
  outputChannel[pixelLocation]=valuesSum;
}

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numPixels,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
  size_t pixelLocation=blockIdx.x*1024+threadIdx.x;
  if(pixelLocation<numPixels) {
    redChannel[pixelLocation]=inputImageRGBA[pixelLocation].x;
    greenChannel[pixelLocation]=inputImageRGBA[pixelLocation].y;
    blueChannel[pixelLocation]=inputImageRGBA[pixelLocation].z;
  }
}

__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numPixels)
{
  size_t pixelLocation=blockIdx.x*1024+threadIdx.x;

  if(pixelLocation>=numPixels)
    return;

  unsigned char red   = redChannel[pixelLocation];
  unsigned char green = greenChannel[pixelLocation];
  unsigned char blue  = blueChannel[pixelLocation];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[pixelLocation] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

  checkCudaErrors(cudaMalloc(&d_filter, sizeof(float) * filterWidth*filterWidth));
  checkCudaErrors(cudaMemcpy(d_filter,h_filter, sizeof(float) * filterWidth*filterWidth,cudaMemcpyHostToDevice));
}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
  const size_t numPixels=numCols*numRows;
  const dim3 blockSize(1024,1,1);
  const dim3 gridSize(numPixels/1024+ numPixels%1024,1,1);

  float testMem[filterWidth*filterWidth];
  checkCudaErrors(cudaMemcpy(testMem,d_filter, sizeof(float) * filterWidth*filterWidth,cudaMemcpyDeviceToHost));

  separateChannels<<<gridSize,blockSize>>>(d_inputImageRGBA,
                      numPixels,
                      d_red,
                      d_green,
                      d_blue);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  const dim3 blockSizeConv(1024,1,1);
  const dim3 gridSizeConv(numRows/1024+numRows%1024,1,1);

  gaussian_blur<<<gridSizeConv,blockSizeConv>>>(d_red, d_redBlurred, numRows, numCols,numPixels, d_filter, filterWidth);
  gaussian_blur<<<gridSizeConv,blockSizeConv>>>(d_green, d_greenBlurred, numRows, numCols,numPixels, d_filter, filterWidth);
  gaussian_blur<<<gridSizeConv,blockSizeConv>>>(d_blue, d_blueBlurred, numRows, numCols,numPixels, d_filter, filterWidth);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred,
                                             d_greenBlurred,
                                             d_blueBlurred,
                                             d_outputImageRGBA,
                                             numPixels);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}

void cleanup() {
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
}
