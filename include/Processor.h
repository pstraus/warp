// paul.straus@gmail.com
// all rights reserved
#pragma once

// file I_ReferenceImage.h

//Includes
#include <warp/I_Processor.h>
#include <memory>

/// \brief abstract class describing what a reference image must do /
class Processor : I_Processor
{
  public:
    Processor() = delete;

    //Constructor consumes unique_ptr of type I_ReferenceImage
    Processor(std::unique_ptr<I_ReferenceImage>&& refImage);

    virtual ~Processor() = default;

    //actual functions
    cv::cuda::GpuMat processNewImage(cv::cuda::GpuMat& newImage) override;

  private:
    std::unique_ptr<I_ReferenceImage> m_ref;
}