// paul.straus@gmail.com
// all rights reserved


// file I_ReferenceImage.h

//Includes
#include <opencv2/core/opencv.hpp>
#include <opencv2/core/cuda.hpp>

/// \brief abstract class describing what a reference image must do / be updated
class I_ReferenceImage
{
  public:
    virtual cv::cuda::GpuMat getReferenceImage() = 0;
    virtual void update(cv::Mat& newImage) = 0;
    virtual bool valid() = 0;
}