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
    virtual ~I_ReferenceImage() = default;
    virtual const cv::cuda::GpuMat& getReferenceImage() const = 0;
    virtual void update(const cv::cuda::GpuMat& newImage) = 0;
    virtual bool valid() const = 0;
};