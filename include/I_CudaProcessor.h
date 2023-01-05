// paul.straus@gmail.com
// all rights reserved
#pragma once


// file I_Processor.h

//Includes
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

/// \brief abstract class describing what a Processor class must do
class I_CudaProcessor
{
  public:
    virtual ~I_CudaProcessor() = default;

    //actual functions
    virtual cv::cuda::GpuMat processNewImage(cv::cuda::GpuMat& newImage) = 0;
};