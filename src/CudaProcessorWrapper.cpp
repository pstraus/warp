// paul.straus@gmail.com
// all rights reserved

// file CudaProcessorWrapper.cpp

//Includes
#include <CudaProcessorWrapper.h>
#include <memory>
#include <stdexcept>


using namespace cv;
//Constructor consumes unique_ptr of type I_ReferenceImage
/// \note: explicitly specify cv::cuda::FarnebackOpticalFlow as there is a name collision with cv::FarnebackOpticalFlow
CudaProcessorWrapper::CudaProcessorWrapper(std::unique_ptr<I_CudaProcessor>&& p_CudaProcessor)
: mp_CudaProcessor{std::move(p_CudaProcessor)}
{
}


//actual functions
cv::Mat CudaProcessorWrapper::processNewImage(cv::Mat& newImage)
{
  //Move image into GPU
  cv::cuda::GpuMat newImage_gpu(newImage);

  //Get GPU return
  auto processedImage_gpu = mp_CudaProcessor->processNewImage(newImage_gpu);

  //Allocate space in main memory
  cv::Mat processedImage_cpu(newImage.size(), newImage.type());
  
  //Move from GPU to main memory
  processedImage_gpu.download(processedImage_cpu);

  return processedImage_cpu;

}

