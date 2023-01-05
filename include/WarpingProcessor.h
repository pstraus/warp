// paul.straus@gmail.com
// all rights reserved
#pragma once

// file I_ReferenceImage.h

//Includes
#include <I_CudaProcessor.h>
#include <memory>
#include <opencv2/cudaoptflow.hpp>
#include <I_ReferenceImage.h>

/// \brief abstract class describing what a reference image must do /
class WarpingProcessor : I_CudaProcessor
{
  public:
    WarpingProcessor() = delete;

    //Constructor consumes unique_ptr of type I_ReferenceImage
    WarpingProcessor(std::unique_ptr<I_ReferenceImage>&& refImage, cv::Ptr<cv::cuda::FarnebackOpticalFlow> flowPtr);

    virtual ~WarpingProcessor() = default;

    //actual functions
    cv::cuda::GpuMat processNewImage(cv::cuda::GpuMat& newImage) override;

  private:
    void genBaseCoordinateMaps(const cv::cuda::GpuMat& image);
    std::unique_ptr<I_ReferenceImage> m_ref;
    cv::Ptr<cv::cuda::FarnebackOpticalFlow> mp_opticalFlow;

    //Holds a map of the x and y coordinates of the pixels in the image to be warped.  This is _always_ the same and only initialized once; we add the optical flow deltas to them
    cv::cuda::GpuMat m_x;
    cv::cuda::GpuMat m_y;
};