// paul.straus@gmail.com
// all rights reserved
#pragma once

// file ExponentialAverage.h

//Includes
#include <I_ReferenceImage.h>

/// \brief concrete implementation of a reference image generator
class ExponentialAverage : public I_ReferenceImage
{
  public:
    ExponentialAverage() = delete;
    ExponentialAverage(double alpha);
    ExponentialAverage(const cv::cuda::GpuMat& firstImage, double alpha) ;

    ~ExponentialAverage() override = default;

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