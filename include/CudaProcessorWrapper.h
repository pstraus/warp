// paul.straus@gmail.com
// all rights reserved
#pragma once


// file CudaProcessorWrapper.h

//Includes
#include <opencv2/opencv.hpp>
#include <I_Processor.h>
#include <I_CudaProcessor.h>

/// \brief abstract class describing what a Processor class must do
class CudaProcessorWrapper : public I_Processor
{
  public:
    CudaProcessorWrapper() = default;
    CudaProcessorWrapper(std::unique_ptr<I_CudaProcessor>&& pCudaProcessor);

    virtual ~CudaProcessorWrapper() = default;

    //actual functions
    cv::Mat processNewImage(cv::Mat& newImage) override;

  private:
    std::unique_ptr<I_CudaProcessor> mp_CudaProcessor;
};