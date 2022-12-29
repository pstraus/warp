// paul.straus@gmail.com
// all rights reserved
#pragma once

// file I_ReferenceImage.h

//Includes
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

/// \brief abstract class describing what a reference image must do / be updated
class I_ReferenceImage
{
  public:
    virtual const cv::cuda::GpuMat& getReferenceImage() = 0;
    virtual void update(const cv::Mat& newImage) = 0;
    virtual bool valid() = 0;
}