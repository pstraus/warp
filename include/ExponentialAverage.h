// paul.straus@gmail.com
// all rights reserved
#pragma once

// file I_ReferenceImage.h

//Includes
#include <I_ReferenceImage.h>

/// \brief abstract class describing what a reference image must do /
class ExponentialAverage : public I_ReferenceImage
{
  public:
    ExponentialAverage() = delete;
    ExponentialAverage(const cv::cuda::GpuMat& firstImage, double alpha) ;

    virtual ~ExponentialAverage() = default;

    //There's some rule of 5 things that should be done here; functionally, we never intend to copy / move this

    //actual functions
    const cv::cuda::GpuMat& getReferenceImage() const override;
    void update(const cv::cuda::GpuMat& newImage) override;
    bool valid() const override;

  private:
    cv::cuda::GpuMat m_ref;
    bool m_initialized = false;
    double m_alpha;
};